"""Read the code and pipeline that produced a result and find the bugs that fake it.

Static only: files are parsed with `ast`, never imported or executed, so it is
safe to point at a stranger's repo. Covers Python files and Jupyter notebooks.
"""
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

from .common import Check, report, PASS, WARN, FAIL

SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "site-packages", "__pycache__",
             "build", "dist", ".tox", ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache"}
MAX_FILES, MAX_BYTES = 3000, 2_000_000

# rule id -> (title, severity, why it fakes results, fix). severity: high -> FAIL, medium/low -> WARN
RULES = {
    "future_shift": ("Uses future values (negative shift)", "high",
                     "`.shift(-n)` pulls a value from n rows in the future. Anywhere other than building the "
                     "target, the model or signal gets to see tomorrow.",
                     "Only use shift(-n) to build the label, and never feed that column back in as a feature."),
    "next_bar_signal": ("Signal peeks at the next bar", "high",
                        "A decision at row i reads row i+1 - something you can't know yet at that moment.",
                        "Decide on data up to i; only use i+1 for the fill price / PnL."),
    "centered_window": ("Centered rolling window", "high",
                        "`rolling(center=True)` averages in future rows, so every value already knows what's next.",
                        "Use a trailing window (the default center=False)."),
    "backfill": ("Back-filling missing values", "medium",
                 "`bfill` copies a future value backwards into the past.",
                 "Forward-fill (ffill) or drop; never fill the past from the future."),
    "full_sample_norm": ("Normalized with whole-history stats", "medium",
                         "Subtracting the mean / dividing by the std of the *whole* series bakes future "
                         "information into every earlier row.",
                         "Use rolling/expanding stats, or stats fitted on the training period only."),
    "fit_before_split": ("Preprocessing fitted before the train/test split", "high",
                         "A scaler/encoder/feature-selector/resampler fitted on all the data learns from the test "
                         "set, so the test score is inflated.",
                         "Split first, fit on train only - or put the step inside an sklearn Pipeline."),
    "shuffled_time_split": ("Time-series data split randomly", "medium",
                            "Shuffling rows before splitting lets the model train on the future and test on the "
                            "past.",
                            "Use shuffle=False / TimeSeriesSplit / a cutoff date."),
    "test_set_selection": ("Best version picked using the test set", "high",
                           "A loop tries options and keeps whichever scores best on the test data. The test set "
                           "is now part of training, and the reported score is the luckiest of many.",
                           "Choose on a separate validation set; touch the test set exactly once at the end."),
    "seed_fishing": ("Random seed picked for the best result", "high",
                     "Trying many seeds and keeping the best one reports luck, not skill.",
                     "Fix the seed up front, or report the average and spread over all seeds."),
    "train_score": ("Scored on training data", "medium",
                    "A metric computed on the data the model trained on says how well it memorized, not how "
                    "well it works.",
                    "Report metrics on held-out data only."),
    "survivorship": ("Hard-coded list of today's winners", "low",
                     "Backtesting on a fixed list of tickers that exist today skips every company that went "
                     "bust or got delisted - survivorship bias.",
                     "Use point-in-time universe membership including delisted names."),
    "no_costs": ("Backtest without trading costs", "medium",
                 "No fees, commissions, spread or slippage anywhere in the backtest code.",
                 "Charge realistic costs per trade - many edges vanish here."),
    "unseeded": ("Results not reproducible (no random seed)", "low",
                 "The code uses randomness but never sets a seed - nobody, including you, can reproduce "
                 "the exact number reported.",
                 "Set and record a seed (np.random.default_rng(seed), random_state=seed)."),
    "notebook_order": ("Notebook run out of order", "medium",
                       "Cells were executed out of order, so the saved outputs may depend on hidden state "
                       "that the code on the page doesn't recreate.",
                       "Restart the kernel and run all cells top to bottom, then re-check the numbers."),
}

LABEL_HINT = re.compile(r"(target|label|^y$|^y_|fwd|forward|future|next|outcome|ret_?\d|return_?\d)", re.I)
DECISION_HINT = re.compile(r"(signal|entry|enter|feature|pred|position|pos$|^x|long|short|buy|sell)", re.I)
PREPROC = {"StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer", "PCA",
           "SelectKBest", "SelectFromModel", "RFE", "VarianceThreshold", "SimpleImputer", "KNNImputer",
           "IterativeImputer", "QuantileTransformer", "PowerTransformer", "TfidfVectorizer",
           "CountVectorizer", "OneHotEncoder", "TargetEncoder", "SMOTE", "ADASYN", "RandomOverSampler",
           "RandomUnderSampler", "KBinsDiscretizer"}
SPLITS = {"train_test_split", "TimeSeriesSplit", "KFold", "StratifiedKFold", "GroupKFold",
          "cross_val_score", "cross_validate", "ShuffleSplit"}
METRICS = {"accuracy_score", "r2_score", "mean_squared_error", "mean_absolute_error", "roc_auc_score",
           "f1_score", "precision_score", "recall_score", "log_loss", "balanced_accuracy_score"}
TIME_HINT = re.compile(r"(datetime|to_datetime|timestamp|\bdate\b|\.shift\(|\.rolling\(|ohlc|close|yfinance|"
                       r"resample\()", re.I)
COST_HINT = re.compile(r"(cost|fee|commission|slippage|spread|bps|tick_value|friction)", re.I)
BACKTEST_HINT = re.compile(r"(backtest|pnl|equity_curve|cum_?ret|sharpe)", re.I)
RANDOM_HINT = re.compile(r"(np\.random\.|random\.(random|choice|shuffle|randint)|\.sample\(|RandomForest|"
                         r"shuffle=True|torch\.rand|default_rng\(\))")
SEED_HINT = re.compile(r"(seed|random_state)", re.I)


def _name(node) -> str:
    """Last name in a call target: df.x.shift -> 'shift', train_test_split -> 'train_test_split'."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _neg_const(node) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return isinstance(node.operand.value, (int, float)) and node.operand.value > 0
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value < 0


def _kw(call: ast.Call, key: str):
    return next((k.value for k in call.keywords if k.arg == key), None)


def _names_in(node) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


class _Auditor(ast.NodeVisitor):
    def __init__(self, source: str, time_series: bool, backtest: bool):
        self.lines = source.splitlines()
        self.time_series = time_series
        self.backtest = backtest
        self.hits = []          # (rule, line)
        self.preproc_vars = set()
        self.fits = []          # (line, arg text) of preprocessing fits
        self.split_lines = []
        self._targets = []

    def hit(self, rule, node):
        self.hits.append((rule, node.lineno))

    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, ast.Name) and isinstance(node.value, ast.Call) and _name(node.value.func) in PREPROC:
                self.preproc_vars.add(t.id)
        self._targets.append(" ".join(ast.unparse(t) for t in node.targets))
        self.generic_visit(node)
        self._targets.pop()

    def visit_Call(self, node):
        fn = _name(node.func)
        recv = node.func.value if isinstance(node.func, ast.Attribute) else None

        if fn == "shift":
            periods = node.args[0] if node.args else _kw(node, "periods")
            if periods is not None and _neg_const(periods):
                target = self._targets[-1] if self._targets else ""
                shifted = ast.unparse(recv) if recv is not None else ""
                is_label = LABEL_HINT.search(target.split("[")[-1].strip("'\"] ")) or LABEL_HINT.search(target)
                is_calendar = re.search(r"date|time|session|day|week|month", shifted, re.I)
                # a forward return used inline to score PnL is fine; stored as a column or signal it isn't
                scores_pnl = re.search(r"ret|return|pnl|close|price", shifted, re.I) and "[" not in target \
                    and not DECISION_HINT.search(target)
                if not (is_label or is_calendar or scores_pnl):
                    self.hit("future_shift", node)
        elif fn == "rolling":
            c = _kw(node, "center")
            if isinstance(c, ast.Constant) and c.value is True:
                self.hit("centered_window", node)
        elif fn in ("bfill", "backfill"):
            self.hit("backfill", node)
        elif fn == "fillna":
            m = _kw(node, "method")
            if isinstance(m, ast.Constant) and m.value in ("bfill", "backfill"):
                self.hit("backfill", node)
        elif fn in SPLITS:
            self.split_lines.append(node.lineno)
            if self.time_series:
                sh = _kw(node, "shuffle")
                if fn == "train_test_split" and not (isinstance(sh, ast.Constant) and sh.value is False):
                    self.hit("shuffled_time_split", node)
                elif fn in ("KFold", "StratifiedKFold", "ShuffleSplit") and not (
                        isinstance(sh, ast.Constant) and sh.value is False and fn != "ShuffleSplit"):
                    self.hit("shuffled_time_split", node)
        elif fn in ("fit", "fit_transform", "fit_resample") and recv is not None:
            is_pre = (isinstance(recv, ast.Name) and recv.id in self.preproc_vars) or \
                     (isinstance(recv, ast.Call) and _name(recv.func) in PREPROC)
            if is_pre and node.args:
                self.fits.append((node.lineno, ast.unparse(node.args[0])))
        elif fn in METRICS or fn == "score":
            args = " ".join(ast.unparse(a) for a in node.args).lower()
            if "train" in args and not re.search(r"test|val|holdout|oos", args):
                self.hit("train_score", node)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        # x - x.mean()  /  (x - x.min()) / (x.max() - x.min())  on the full series
        if self.time_series and isinstance(node.op, (ast.Sub, ast.Div)):
            right = node.right
            if isinstance(right, ast.Call) and _name(right.func) in ("mean", "std", "min", "max") \
                    and not right.args and isinstance(right.func, ast.Attribute):
                base = ast.unparse(right.func.value)
                target = self._targets[-1] if self._targets else ""
                builds_input = "[" in target or DECISION_HINT.search(target)  # a column/feature, not a stat
                left_is_stat = isinstance(node.left, ast.Call) and _name(node.left.func) in ("mean", "std", "min", "max")
                if builds_input and not left_is_stat and \
                        not re.search(r"roll|expanding|ewm|window|train|groupby|axis", base + ast.unparse(right)):
                    left = ast.unparse(node.left)
                    if base and base in left:
                        self.hit("full_sample_norm", node)
        self.generic_visit(node)

    def _loop(self, node):
        body = ast.Module(body=node.body, type_ignores=[])
        src = ast.unparse(body)
        tracks_best = bool(re.search(r"\bbest\w*\s*=|>\s*best|best\w*\s*<|max\(|argmax|idxmax", src))
        loop_var = ast.unparse(node.target) if isinstance(node, ast.For) else ""
        if tracks_best:
            calls = [n for n in ast.walk(body) if isinstance(n, ast.Call)]
            on_test = any(re.search(r"test", ast.unparse(a), re.I)
                          for c in calls if _name(c.func) in METRICS | {"score", "predict", "evaluate",
                                                                         "predict_proba", "backtest"}
                          for a in c.args)
            if on_test:
                self.hit("test_set_selection", node)
            if re.search(r"seed|state", loop_var, re.I) and re.search(
                    rf"(random_state|seed)\s*=\s*{re.escape(loop_var)}|seed\({re.escape(loop_var)}\)", src):
                self.hit("seed_fishing", node)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            i = node.target.id
            for n in ast.walk(body):
                if isinstance(n, (ast.If, ast.Assign)):
                    zone = n.test if isinstance(n, ast.If) else n.value
                    if isinstance(n, ast.Assign) and not DECISION_HINT.search(
                            " ".join(ast.unparse(t) for t in n.targets)):
                        continue
                    for s in ast.walk(zone):
                        if isinstance(s, ast.Subscript) and isinstance(s.slice, ast.BinOp) \
                                and isinstance(s.slice.op, ast.Add) and _names_in(s.slice.left) == {i} \
                                and isinstance(s.slice.right, ast.Constant):
                            self.hit("next_bar_signal", s)
                            break
        self.generic_visit(node)

    visit_For = visit_While = _loop

    def visit_List(self, node):
        vals = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if self.backtest and len(vals) >= 5 and len(vals) == len(node.elts) and all(re.fullmatch(r"[A-Z]{1,5}", v) for v in vals):
            self.hit("survivorship", node)
        self.generic_visit(node)

    def finish(self):
        if self.split_lines:
            first_split = min(self.split_lines)
            for line, arg in self.fits:
                if line < first_split and not re.search(r"train", arg, re.I):
                    self.hits.append(("fit_before_split", line))
        return self.hits


def _notebook_cells(path: Path) -> tuple[list[str], bool]:
    """Code cells (audited one by one, so one broken cell doesn't hide the rest) + out-of-order flag."""
    nb = json.loads(path.read_text(errors="replace"))
    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    counts = [c.get("execution_count") for c in cells if c.get("execution_count")]
    out_of_order = any(b < a for a, b in zip(counts, counts[1:]))
    src = []
    for c in cells:
        text = "".join(c.get("source", []))
        # drop IPython magics / shell lines, which aren't Python
        src.append("\n".join("" if l.lstrip().startswith(("%", "!")) else l for l in text.splitlines()))
    return src, out_of_order


def iter_files(root: Path):
    n = 0
    for p in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in p.parts) or not p.is_file():
            continue
        if p.suffix in (".py", ".ipynb") and p.stat().st_size <= MAX_BYTES:
            yield p
            n += 1
            if n >= MAX_FILES:
                return


def audit_source(source: str, filename: str = "<code>", context: str | None = None) -> list[tuple[str, str, int, str]]:
    """Audit one source string. Returns (rule, file, line, code) findings.
    context: surrounding text (e.g. the whole notebook) used to decide whether this is time-series / backtest code."""
    tree = ast.parse(source)
    ctx = context if context is not None else source
    a = _Auditor(source, bool(TIME_HINT.search(ctx)), bool(BACKTEST_HINT.search(ctx)))
    a.visit(tree)
    lines = source.splitlines()
    return [(r, filename, ln, lines[ln - 1].strip()[:160] if 0 < ln <= len(lines) else "")
            for r, ln in a.finish()]


def run(root: str | Path) -> dict:
    root = Path(root)
    paths = [root] if root.is_file() else list(iter_files(root))
    findings, unreadable = [], []
    corpus = []
    for p in paths:
        rel = str(p.relative_to(root)) if p != root else p.name
        try:
            if p.suffix == ".ipynb":
                cells, ooo = _notebook_cells(p)
                if ooo:
                    findings.append(("notebook_order", rel, 1, ""))
                whole = "\n".join(cells)
                corpus.append(whole)
                bad = 0
                for i, cell in enumerate(cells, 1):
                    try:
                        findings += audit_source(cell, f"{rel} [cell {i}]", context=whole)
                    except SyntaxError:
                        bad += 1
                if bad == len(cells) and cells:
                    unreadable.append(rel)
                continue
            src = p.read_text(errors="replace")
            corpus.append(src)
            findings += audit_source(src, rel)
        except (SyntaxError, ValueError, json.JSONDecodeError):
            unreadable.append(rel)

    text = "\n".join(corpus)
    is_backtest = bool(BACKTEST_HINT.search(text))
    if is_backtest and not COST_HINT.search(text):
        findings.append(("no_costs", "(whole project)", 0, ""))
    if RANDOM_HINT.search(text) and not SEED_HINT.search(text):
        findings.append(("unseeded", "(whole project)", 0, ""))

    by_rule = defaultdict(list)
    for r, f, ln, code in findings:
        by_rule[r].append({"file": f, "line": ln, "code": code})

    checks = []
    for rule, (title, sev, why, fix) in RULES.items():
        locs = by_rule.get(rule)
        if not locs:
            checks.append(Check(title, PASS))
            continue
        where = "; ".join(f"{l['file']}:{l['line']}" if l["line"] else l["file"] for l in locs[:5])
        more = f" (+{len(locs) - 5} more)" if len(locs) > 5 else ""
        checks.append(Check(title, FAIL if sev == "high" else WARN, f"{why} Found at {where}{more}.", fix,
                            {"severity": sev, "locations": locs}))
    if unreadable:
        checks.append(Check("All files readable", WARN,
                            f"Couldn't parse {len(unreadable)} file(s): {', '.join(unreadable[:5])}",
                            "Fix syntax errors so they can be audited."))
    return report("Code & pipeline audit", checks, n=len(paths), unit="files")
