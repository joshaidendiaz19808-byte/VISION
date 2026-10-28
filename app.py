import hashlib
import io
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, "src")
from checks import ab_test, ai_review, code_audit, connectors, detect, integrity, ml_model, trading, merge, FAIL, WARN, PASS

# ── helpers ───────────────────────────────────────────────────────────────────

KINDS = {
    "Auto-detect": "auto",
    "Trading strategy": "trading",
    "A/B test / experiment": "ab",
    "ML / AI model": "ml",
    "Any data / research numbers": "data",
    "Code & pipeline only": "code",
}
LOOKS_LIKE = {"trading": "trading results", "ab": "an A/B test", "predictions": "model predictions",
              "data": "general data"}
COLORS = {"LOOKS REAL": "#1D9E75", "QUESTIONABLE": "#E5A50A", "LOOKS FAKE": "#E5484D",
          "NOT ENOUGH TO JUDGE": "#6B7280"}
ICON = {FAIL: ("✗", "#E5484D"), WARN: ("!", "#E5A50A")}
NONE = "(none)"


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() > len(df) * 0.5]


def is_dollar_scale(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return len(vals) > 0 and vals.abs().median() > 10


def read_table(f) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parsed table + the same table as raw strings (integrity checks need the digits as typed)."""
    data = f.getvalue()
    reader = pd.read_excel if f.name.endswith((".xlsx", ".xls")) else pd.read_csv
    df, raw = reader(io.BytesIO(data)), reader(io.BytesIO(data), dtype=str)
    df.columns = raw.columns = [str(c).strip() for c in df.columns]
    return df, raw


MAX_UNZIPPED = 200 * 1024 * 1024


def code_dir_from_uploads(files) -> Path | None:
    """Put uploaded .py / .ipynb / .zip files in a temp folder for the static audit (never executed).
    Cached by content, so reruns reuse the same folder (and the AI review stays attached to it)."""
    digest = hashlib.sha256(b"".join(f.name.encode() + f.getvalue() for f in files)).hexdigest()
    cache = st.session_state.setdefault("uploads", {})
    if digest in cache and Path(cache[digest]).exists():
        return Path(cache[digest])
    d = Path(tempfile.mkdtemp(prefix="vision_code_"))
    for f in files:
        if f.name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z:
                if sum(i.file_size for i in z.infolist()) > MAX_UNZIPPED:
                    st.error(f"{f.name} unpacks to more than 200 MB - upload just the code folder.")
                    shutil.rmtree(d, ignore_errors=True)
                    return None
                z.extractall(d)  # zipfile strips absolute paths and '..'
        else:
            (d / Path(f.name).name).write_bytes(f.getvalue())
    cache[digest] = str(d)
    return d


def sweep_old_temp(max_age_s=3600):
    """Hosted servers live a long time: drop downloaded/uploaded code older than an hour."""
    for p in Path(tempfile.gettempdir()).glob("vision_*"):
        try:
            if time.time() - p.stat().st_mtime > max_age_s:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


def auth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:  # no secrets file at all = local use
        return False


def pick(label, options, guess, key, optional=False):
    options = ([NONE] if optional else []) + list(options)
    choice = st.selectbox(label, options, index=options.index(guess) if guess in options else 0, key=key)
    return None if choice == NONE else choice


def fetch(fn, *args):
    try:
        return fn(*args)
    except Exception as e:  # connector errors are already written for humans and never contain the key
        st.error(str(e))


def mlflow_runs_block():
    """Optional: pull every run from MLflow / Databricks, so 'versions tried' is counted, not guessed."""
    with st.expander("Count the versions you tried from MLflow / Databricks (optional)"):
        c1, c2 = st.columns(2)
        host = c1.text_input("Tracking server URL", placeholder="https://dbc-....cloud.databricks.com")
        exp = c2.text_input("Experiment ID(s), comma-separated")
        tok = st.text_input("Token (if the server needs one)", type="password", key="mlflow_tok")
        if host and exp and st.button("Pull runs"):
            st.session_state.mlflow = fetch(connectors.mlflow_runs, host, [e.strip() for e in exp.split(",")],
                                            tok or None)
        runs = st.session_state.get("mlflow")
        if runs is not None:
            st.caption(f"{len(runs)} runs found.")
        return runs


def show_report(rep: dict):
    color = COLORS[rep["verdict"]]
    sub = f"{rep['n']} {rep['unit']} analyzed · " if rep.get("n") else ""
    st.markdown(
        f"""
        <div style="background:#1A1D24;border:1px solid {color}33;border-left:4px solid {color};
                    border-radius:10px;padding:28px 32px;margin:8px 0 24px 0;">
          <div style="display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:12px;">
            <span style="font-size:2.1rem;font-weight:800;color:{color};letter-spacing:0.02em;">{rep['verdict']}</span>
            <span style="background:{color}22;color:{color};font-weight:700;font-size:0.85rem;padding:6px 14px;
                         border-radius:999px;border:1px solid {color}55;">{rep['score']}</span>
          </div>
          <div style="color:#6B7280;font-size:0.85rem;margin-top:10px;">{sub}{rep['kind']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    problems = sorted((c for c in rep["checks"] if c.status in (FAIL, WARN)), key=lambda c: c.status != FAIL)
    if problems:
        st.markdown("<div style='font-weight:700;font-size:1.1rem;margin-bottom:8px;'>What's wrong</div>",
                    unsafe_allow_html=True)
    for c in problems:
        mark, col = ICON[c.status]
        st.markdown(
            f"""
            <div style="background:#1A1D24;border-radius:8px;padding:16px 20px;margin-bottom:12px;">
              <div style="font-weight:700;color:#FAFAFA;">
                <span style="color:{col};margin-right:8px;">{mark}</span>{c.name}</div>
              <div style="font-size:0.95rem;color:#D1D5DB;margin-top:6px;padding-left:24px;">{c.reason}</div>
              <div style="font-size:0.9rem;color:#9CA3AF;margin-top:6px;padding-left:24px;">Fix: {c.fix}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        locs = c.detail.get("locations")
        if locs and len(locs) > 1:
            with st.expander(f"All {len(locs)} places"):
                st.code("\n".join(f"{l['file']}:{l['line']}  {l['code']}" for l in locs), language="text")
    passed = [c for c in rep["checks"] if c.status == PASS]
    skipped = [c for c in rep["checks"] if c.status == "skip"]
    if not problems and passed:
        st.success("Passed every check VISION could run. Still confirm it on fresh data before betting on it.")
    with st.expander(f"Passed ({len(passed)}) · Couldn't run ({len(skipped)})"):
        for c in passed:
            st.markdown(f"✓ {c.name}")
        for c in skipped:
            st.markdown(f"– **{c.name}** — {c.reason}")


# ── page ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="VISION", page_icon="🔍")

# Hosted = a login provider is configured in secrets. Locally there's no login.
HOSTED = auth_configured()
if HOSTED:
    if not st.user.get("is_logged_in", False):
        st.markdown("<div style='font-size:2.4rem;font-weight:800;color:#FAFAFA;'>VISION</div>"
                    "<div style='color:#9CA3AF;margin-bottom:1.5rem;'>Are your results real? Log in to check.</div>",
                    unsafe_allow_html=True)
        st.button("Log in", on_click=st.login, type="primary")
        st.stop()
    _, right = st.columns([5, 1])
    right.button("Log out", on_click=st.logout)
    sweep_old_temp()
st.markdown(
    """
    <div style="font-size:2.4rem;font-weight:800;letter-spacing:0.03em;color:#FAFAFA;margin-bottom:0.1rem;">VISION</div>
    <div style="font-size:1rem;color:#9CA3AF;margin-bottom:1.5rem;">
        Are your results real? Plug in what you're testing - trades, an experiment, a model,
        any data, or the code behind it - and get the truth.
    </div>
    """,
    unsafe_allow_html=True,
)

kind = KINDS[st.radio("What are you checking?", list(KINDS), horizontal=True)]
reports = []

# ── results: upload or connect ────────────────────────────────────────────────

if kind == "ml":
    st.caption("Upload your **train** and **test** tables to check for leakage, and/or a **predictions** "
               "table (actual vs predicted) to check the score.")
    c1, c2, c3 = st.columns(3)
    tr = c1.file_uploader("Train table", type=["csv", "xlsx"])
    te = c2.file_uploader("Test table", type=["csv", "xlsx"])
    pr = c3.file_uploader("Predictions", type=["csv", "xlsx"])
    train = read_table(tr)[0] if tr else None
    test = read_table(te)[0] if te else None
    pred = read_table(pr)[0] if pr else None
    args = {}
    if train is not None and test is not None:
        cols = list(train.columns)
        c1, c2, c3 = st.columns(3)
        with c1:
            target = pick("Target column", cols, detect.find(train, detect.YTRUE), "ml_t")
        with c2:
            tcol = pick("Time column", cols, detect.find(train, detect.DATE), "ml_d", optional=True)
        with c3:
            icol = pick("Entity / user ID column", cols,
                        next((c for c in cols if re.search(r"id$", c, re.I)), None), "ml_i", optional=True)
        args.update(train=train, test=test, target=target, time_col=tcol, id_col=icol)
    if pred is not None:
        c1, c2 = st.columns(2)
        with c1:
            yt = pick("Actual column", pred.columns, detect.find(pred, detect.YTRUE), "ml_yt")
        with c2:
            yp = pick("Predicted column", pred.columns, detect.find(pred, detect.YPRED), "ml_yp")
        args.update(pred=pred, y_true=yt, y_pred=yp)
    runs = mlflow_runs_block()
    metric_cols = [c for c in runs.columns if c.startswith("metric.")] if runs is not None else []
    if metric_cols:
        c1, c2, c3 = st.columns(3)
        with c1:
            mcol = pick("Metric that picked the winner", metric_cols, None, "ml_m")
        n_test = c2.number_input("Test-set size", 1, 10_000_000, len(test) if test is not None else 1000)
        higher = c3.toggle("Higher is better", value=True)
        args.update(run_scores=runs[mcol], n_test=int(n_test), higher_is_better=higher)
    if args:
        reports.append(ml_model.run(**args))

elif kind != "code":
    source = st.radio("Where are the results?", ["Upload a file", "Connect an account (API)"], horizontal=True)
    df = raw = None
    if source == "Upload a file":
        f = st.file_uploader("CSV or Excel - any layout, VISION figures it out", type=["csv", "xlsx", "xls"])
        if f:
            try:
                df, raw = read_table(f)
            except Exception as e:
                st.error(f"Couldn't read that file: {e}")
                st.stop()
        elif kind in ("auto", "trading"):
            if st.button("▶ Try a sample"):
                st.session_state.sample = True
            if st.session_state.get("sample"):
                df = pd.read_csv("trades_sample.csv")
                raw = df.astype(str)
    else:
        brokers = ["Interactive Brokers (Flex Query)", "Alpaca (broker)"]
        providers = brokers + ["Any API / URL"] if kind in ("auto", "trading") else ["Any API / URL"]
        prov = st.selectbox("Connect to", providers)
        st.caption("🔒 Keys stay in memory for this session only - never saved, logged, or sent anywhere "
                   "except the provider you pick.")
        if prov.startswith("Interactive"):
            st.caption("One-time setup in IBKR Client Portal: Reports → Flex Queries → new Activity query with "
                       "the Trades section (include FIFO P/L Realized + Cost Basis), then Flex Web Service → "
                       "get a token.")
            c1, c2 = st.columns(2)
            tok = c1.text_input("Flex token", type="password")
            qid = c2.text_input("Query ID")
            if tok and qid and st.button("Connect"):
                with st.spinner("IBKR is building your statement (can take ~30s)…"):
                    st.session_state.fetched = fetch(connectors.ibkr_flex_trades, tok, qid)
        elif prov.startswith("Alpaca"):
            c1, c2 = st.columns(2)
            kid = c1.text_input("API key ID")
            sec = c2.text_input("Secret key", type="password")
            paper = st.toggle("Paper account", value=True)
            if kid and sec and st.button("Connect"):
                st.session_state.fetched = fetch(connectors.alpaca_returns, kid, sec, paper)
        else:
            url = st.text_input("URL that returns CSV or JSON")
            c1, c2 = st.columns(2)
            hdr = c1.text_input("Auth header name (optional)", placeholder="Authorization")
            key = c2.text_input("Key / token", type="password", placeholder="Bearer ...")
            jp = st.text_input("JSON path to the rows (optional)", placeholder="data.results")
            if url and st.button("Fetch"):
                st.session_state.fetched = fetch(connectors.url_table, url, hdr or None, key or None, jp or None)
        df = st.session_state.get("fetched")
        raw = df.astype(str) if df is not None else None

    if df is not None:
        if df.empty:
            st.error("That table is empty.")
            st.stop()
        guessed, cols = detect.guess(df)
        if kind == "auto":
            kind = guessed
            st.caption(f"Looks like **{LOOKS_LIKE[kind]}** - pick a type above if that's wrong.")
        allc = list(df.columns)

        if kind == "trading":
            nums = numeric_columns(df)
            if not nums:
                st.error("No numeric columns found - VISION needs a % return per trade.")
                st.stop()
            c1, c2, c3 = st.columns(3)
            with c1:
                rcol = pick("% return column", nums, cols.get("returns") or detect.find(df, detect.RETURN), "t_r")
            cost = c2.number_input("Cost per trade (bps)", 0.0, 100.0, 5.0)
            runs = st.session_state.get("mlflow")
            tries = c3.number_input("Versions you tried", 1, 100000, len(runs) if runs is not None else 1,
                                    help="How many settings/variants did you test before picking this one?")
            if is_dollar_scale(df[rcol]):
                st.error(f"'{rcol}' looks like dollar amounts, not % returns (0.5 = a 0.5% gain). "
                         "Pick your % column, or convert first.")
                st.stop()
            mlflow_runs_block()
            dcol = cols.get("date") or detect.find(df, detect.DATE)
            ordered = df.sort_values(dcol) if dcol else df
            reports.append(trading.run(ordered[rcol], cost_bps=cost, n_trials=int(tries)))

        elif kind == "ab":
            c1, c2, c3 = st.columns(3)
            with c1:
                g = pick("Group column", allc, cols.get("group") or detect.find(df, detect.GROUP), "ab_g")
            with c2:
                o = pick("Outcome column", allc, cols.get("outcome") or detect.find(df, detect.OUTCOME), "ab_o")
            with c3:
                t = pick("Time column", allc, cols.get("time") or detect.find(df, detect.DATE), "ab_t", optional=True)
            c1, c2, c3 = st.columns(3)
            groups = sorted(df[g].astype(str).unique())
            ctrl = c1.selectbox("Control group", groups,
                                index=next((i for i, x in enumerate(groups)
                                            if x.lower() in ("control", "a", "0", "baseline")), 0))
            split = c2.number_input("Planned % in control", 1, 99, 50) / 100
            metrics = c3.number_input("Metrics you looked at", 1, 1000, 1,
                                      help="Checked 20 metrics and reporting the one that moved? Say 20.")
            reports.append(ab_test.run(df, g, o, ctrl, t, split, int(metrics)))

        elif kind == "predictions":
            c1, c2 = st.columns(2)
            with c1:
                yt = pick("Actual column", allc, cols.get("y_true"), "p_yt")
            with c2:
                yp = pick("Predicted column", allc, cols.get("y_pred"), "p_yp")
            reports.append(ml_model.run(pred=df, y_true=yt, y_pred=yp))

        else:
            reports.append(integrity.run(df, raw))

# ── the code behind it ────────────────────────────────────────────────────────

st.markdown("---")
title = "The code / pipeline to audit" if kind == "code" else "Also audit the code that produced this (recommended)"
with st.expander(title, expanded=kind == "code"):
    st.caption("VISION reads the code - it never runs it. Catches look-ahead, leakage, test-set tuning, "
               "seed fishing, missing costs and more, in Python files and notebooks.")
    c1, c2 = st.columns(2)
    files = c1.file_uploader(".py / .ipynb / .zip", type=["py", "ipynb", "zip"], accept_multiple_files=True)
    repo = c2.text_input("…or a GitHub repo", placeholder="owner/repo or https://github.com/owner/repo")
    tok = c2.text_input("GitHub token (private repos only)", type="password")
    code_root = None
    if files:
        code_root = code_dir_from_uploads(files)
    elif repo:
        repos = st.session_state.setdefault("repos", {})
        if repo not in repos and st.button("Audit repo"):
            with st.spinner("Downloading and reading the code…"):
                got = fetch(connectors.github_repo, repo, tok or None)
                if got:
                    repos[repo] = str(got)
        code_root = repos.get(repo)
    if code_root:
        static = code_audit.run(code_root)
        reports.append(static)
        st.markdown("**Deeper AI review** - Claude reads the code like a skeptical reviewer and finds what "
                    "pattern rules can't (leaks through joins, wrong metrics, labels built from the future "
                    "elsewhere in the project). Your code is sent to the Anthropic API; it is never run.")
        server_key_ok = not HOSTED or st.secrets.get("allow_server_ai_key", False)
        akey = st.text_input("Anthropic API key" + (" (blank = use the server's)" if server_key_ok else ""),
                             type="password")
        reviews = st.session_state.setdefault("ai_reviews", {})
        if st.button("Run AI review", disabled=not (akey or server_key_ok)):
            with st.spinner("Claude is reading the code - large projects take a few minutes…"):
                reviews[str(code_root)] = ai_review.run(code_root, akey or None, static)
        ai = reviews.get(str(code_root))
        if ai:
            if ai.get("summary"):
                st.info(ai["summary"])
            reports.append(ai)

# ── verdict ───────────────────────────────────────────────────────────────────

if reports:
    st.markdown("---")
    show_report(reports[0] if len(reports) == 1 else merge("Results + the code behind them", reports))
