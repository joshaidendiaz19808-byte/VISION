"""ML / AI models: is the score real, or did the test set leak into training?

Inputs: the train and test tables (features + target), and/or a predictions
table with y_true and y_pred (class, probability or number).
"""
import numpy as np
import pandas as pd
from scipy import stats

from .common import Check, report, PASS, WARN, FAIL, SKIP


def _is_classification(y: pd.Series) -> bool:
    y = y.dropna()
    return y.dtype == object or y.nunique() <= 20 and np.allclose(y, np.round(y))


def _single_feature_power(x: pd.Series, y: pd.Series, classification: bool) -> float:
    """How well one column alone predicts the target: AUC-style for classes, |corr| for numbers."""
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    if len(x) < 20 or x.nunique() < 2:
        return 0.0
    if not pd.api.types.is_numeric_dtype(x):
        # categorical feature: how pure is the target within each value?
        if not classification or x.nunique() >= len(x) * 0.5:
            return 0.0
        purity = y.groupby(x).agg(lambda s: s.value_counts(normalize=True).iloc[0]).mul(
            x.value_counts(normalize=True)).sum()
        base = y.value_counts(normalize=True).iloc[0]
        # scale so "always guess the majority class" scores 0.5 and perfect scores 1.0
        return float(0.5 + 0.5 * (purity - base) / (1 - base)) if base < 1 else 0.0
    if classification and y.nunique() == 2:
        pos = y == sorted(y.unique())[1]
        auc = stats.mannwhitneyu(x[pos], x[~pos]).statistic / (pos.sum() * (~pos).sum())
        return float(max(auc, 1 - auc))
    if classification:
        return 0.0
    r = np.corrcoef(x.astype(float), y.astype(float))[0, 1]
    return float(abs(r)) if np.isfinite(r) else 0.0


def leakage(train: pd.DataFrame, test: pd.DataFrame, target: str,
            time_col: str | None = None, id_col: str | None = None) -> list[Check]:
    checks = []
    feats = [c for c in train.columns if c in test.columns and c not in (target, time_col, id_col)]

    # Identical rows in train and test: the model is graded on questions it has seen the answers to.
    key = lambda d: pd.util.hash_pandas_object(d[feats].astype(str), index=False)
    overlap = float(key(test).isin(set(key(train))).mean()) if feats else 0.0
    bad = overlap > 0.01
    checks.append(Check("Test rows not in training", FAIL if bad else PASS,
                        f"{overlap:.1%} of test rows also appear in the training data. The model is being "
                        "graded on examples it memorized." if bad else "",
                        "De-duplicate before splitting, then re-score." if bad else "",
                        {"overlap": overlap}))

    if id_col and id_col in train and id_col in test:
        shared = float(test[id_col].isin(set(train[id_col])).mean())
        bad = shared > 0.0
        checks.append(Check("Same people/things not in both sets", FAIL if bad else PASS,
                            f"{shared:.1%} of test rows belong to a '{id_col}' that's also in training. The model "
                            "can recognize the entity instead of learning the pattern." if bad else "",
                            "Split by group (GroupKFold / GroupShuffleSplit) so each entity lands on one side."
                            if bad else ""))

    if time_col and time_col in train and time_col in test:
        tr = pd.to_datetime(train[time_col], errors="coerce")
        te = pd.to_datetime(test[time_col], errors="coerce")
        bad = te.min() < tr.max()
        checks.append(Check("Tested on the future, not the past", FAIL if bad else PASS,
                            f"Training data runs to {tr.max():%Y-%m-%d} but test data starts at {te.min():%Y-%m-%d}. "
                            "The model was trained on information from after the period it's tested on." if bad else "",
                            "Train on everything before a cutoff date, test on everything after it." if bad else ""))

    classification = _is_classification(train[target])
    leaks = []
    for f in feats:
        power = _single_feature_power(train[f], train[target], classification)
        if power > 0.98:
            leaks.append((f, power))
    checks.append(Check("No column gives away the answer", FAIL if leaks else PASS,
                        "These columns predict the target almost perfectly on their own: "
                        + ", ".join(f"'{f}' ({p:.3f})" for f, p in leaks)
                        + ". That's almost always the answer leaking in (a post-outcome field, an ID, or the "
                        "target in disguise)." if leaks else "",
                        "Ask for each one: would I actually know this value at prediction time? If not, drop it."
                        if leaks else "",
                        {"suspects": dict(leaks)}))
    return checks


def _metric(y, p, classification):
    if classification:
        if p.dtype.kind == "f" and np.unique(p).size > 2 and np.unique(y).size == 2:
            return float(np.mean((p >= 0.5).astype(int) == y)), "accuracy"
        return float(np.mean(p == y)), "accuracy"
    ss = ((y - y.mean()) ** 2).sum()
    return (float(1 - ((y - p) ** 2).sum() / ss) if ss else 0.0), "R²"


def predictions(pred: pd.DataFrame, y_true: str, y_pred: str, n_boot: int = 2000, seed: int = 0) -> list[Check]:
    d = pred[[y_true, y_pred]].dropna()
    y = d[y_true].to_numpy()
    p = d[y_pred].to_numpy()
    classification = _is_classification(d[y_true])
    if classification:
        y = pd.factorize(y)[0] if y.dtype == object else y.astype(int)
        p = p if p.dtype.kind == "f" else (pd.Categorical(d[y_pred], categories=pd.unique(d[y_true])).codes
                                           if d[y_pred].dtype == object else p.astype(int))
    score, name = _metric(y, p, classification)
    baseline = float(pd.Series(y).value_counts(normalize=True).iloc[0]) if classification else 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    boots = np.array([_metric(y[i], p[i], classification)[0] for i in idx])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    checks = []

    small = len(y) < 200
    checks.append(Check("Test set big enough", WARN if small else PASS,
                        f"Only {len(y)} test examples - the score could easily be off by "
                        f"{(hi - lo) / 2:.1%}." if small else "",
                        "Score on a bigger held-out set." if small else "", {"n_test": len(y)}))

    beats = lo > baseline
    base_txt = (f"just guessing the most common answer ({baseline:.1%})" if classification
                else "predicting the average (R² 0)")
    checks.append(Check("Better than a dumb baseline", PASS if beats else FAIL,
                        "" if beats else f"{name} is {score:.3f} (95% range {lo:.3f}-{hi:.3f}), which isn't reliably "
                        f"better than {base_txt}.",
                        "" if beats else "The model hasn't learned anything usable yet. Check features and labels.",
                        {"score": score, "ci": (float(lo), float(hi)), "baseline": baseline, "metric": name}))

    too_good = score > 0.99
    checks.append(Check("Not suspiciously perfect", WARN if too_good else PASS,
                        f"{name} = {score:.4f}. Real-world problems almost never score this high - this is "
                        "the classic signature of leakage." if too_good else "",
                        "Run the leakage checks with your train and test tables." if too_good else ""))
    return checks


def best_of_runs(scores: pd.Series, n_test: int, higher_is_better: bool = True) -> Check:
    """Many runs scored on the same test set: how much of the best score is just the luckiest draw?
    scores: one 0-1 metric (accuracy, AUC, F1...) per run, e.g. pulled from MLflow."""
    name = "Best of many runs"
    s = pd.to_numeric(scores, errors="coerce").dropna()
    if len(s) < 2 or not s.between(0, 1).all() or n_test < 1:
        return Check(name, SKIP, "Needs 2+ runs with a 0-1 metric and the test-set size.")
    best = s.max() if higher_is_better else s.min()
    typical = s.median()
    se = float(np.sqrt(max(typical * (1 - typical), 1e-6) / n_test))  # chance wobble of one score
    optimism = se * np.sqrt(2 * np.log(len(s)))  # expected max of N noisy scores above their mean
    gap = abs(best - typical)
    detail = {"runs": len(s), "best": float(best), "median": float(typical), "se": se,
              "expected_luck": float(optimism)}
    if gap <= optimism:
        return Check(name, FAIL,
                     f"You tried {len(s)} versions. The best scored {best:.3f} vs a typical {typical:.3f}, but with "
                     f"{n_test} test examples, picking the best of {len(s)} adds about {optimism:.3f} by luck alone. "
                     "The 'winner' isn't really better than the rest.",
                     "Pick the version on a validation set, then score it once on a fresh test set.", detail)
    return Check(name, WARN,
                 f"The best of {len(s)} runs ({best:.3f}) is genuinely ahead of the pack, but about {optimism:.3f} of "
                 "it is expected luck from picking the max - expect a lower score on new data.",
                 f"Quote roughly {best - optimism if higher_is_better else best + optimism:.3f}, or re-test on "
                 "untouched data.", detail)


def run(train=None, test=None, target=None, pred=None, y_true=None, y_pred=None,
        time_col=None, id_col=None, run_scores=None, n_test=None, higher_is_better=True) -> dict:
    checks = []
    if run_scores is not None and n_test:
        checks.append(best_of_runs(run_scores, n_test, higher_is_better))
    if train is not None and test is not None and target:
        checks += leakage(train, test, target, time_col, id_col)
    if pred is not None and y_true and y_pred:
        checks += predictions(pred, y_true, y_pred)
    if not checks:
        checks.append(Check("Inputs", SKIP, "Give VISION a train + test table, or a predictions table."))
    n = len(test) if test is not None else (len(pred) if pred is not None else None)
    return report("ML / AI model", checks, n=n, unit="test rows")
