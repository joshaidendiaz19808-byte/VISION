"""A/B tests and experiments: is the "winner" a real effect or an artifact?

Input: one row per user/unit with a group column (control vs variant) and an
outcome column (0/1 conversion or a number like revenue). Optional timestamp.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .common import Check, report, PASS, WARN, FAIL, SKIP


def _compare(a: np.ndarray, b: np.ndarray, binary: bool) -> tuple[float, float]:
    """Return (relative lift of b over a, two-sided p-value)."""
    if binary:
        table = [[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]]
        p = stats.chi2_contingency(table, correction=False)[1] if min(a.sum(), b.sum()) > 0 else 1.0
    else:
        p = stats.ttest_ind(a, b, equal_var=False).pvalue
    lift = (b.mean() - a.mean()) / abs(a.mean()) if a.mean() != 0 else np.inf
    return float(lift), float(p)


def run(df: pd.DataFrame, group_col: str, outcome_col: str, control=None, time_col: str | None = None,
        expected_split: float = 0.5, n_metrics: int = 1, alpha: float = 0.05) -> dict:
    df = df[[c for c in (group_col, outcome_col, time_col) if c]].dropna().copy()
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")
    df = df.dropna(subset=[outcome_col])
    groups = list(df[group_col].astype(str).value_counts().index)
    if len(groups) != 2:
        return report("A/B test", [Check("Two groups", FAIL,
                      f"Expected exactly 2 groups in '{group_col}', found {len(groups)}: {groups[:6]}",
                      "Compare one variant against control at a time.")])
    control = str(control) if control is not None else next(
        (g for g in groups if g.lower() in ("control", "a", "0", "baseline")), sorted(groups)[0])
    variant = next(g for g in groups if g != control)
    a = df.loc[df[group_col].astype(str) == control, outcome_col].to_numpy(float)
    b = df.loc[df[group_col].astype(str) == variant, outcome_col].to_numpy(float)
    binary = set(np.unique(df[outcome_col])) <= {0.0, 1.0}
    lift, p = _compare(a, b, binary)
    checks = []

    # 1. Sample ratio mismatch - broken randomization makes every other number meaningless.
    n = len(a) + len(b)
    srm_p = stats.chisquare([len(a), len(b)], [n * expected_split, n * (1 - expected_split)]).pvalue
    checks.append(Check("Groups split as planned", FAIL if srm_p < 0.001 else PASS,
                        f"Planned a {expected_split:.0%}/{1-expected_split:.0%} split but got {len(a)} vs {len(b)} "
                        f"(chance this happens by accident: {srm_p:.1e}). Something is dropping or misrouting users."
                        if srm_p < 0.001 else "",
                        "Find the bug (redirects, bots, logging, eligibility rules) before trusting any result."
                        if srm_p < 0.001 else "", {"srm_p": float(srm_p)}))

    # 2. Enough data.
    events = min(a.sum(), b.sum()) if binary else min(len(a), len(b))
    small = events < (100 if binary else 200)
    checks.append(Check("Enough data", WARN if small else PASS,
                        f"Only {int(events)} {'conversions' if binary else 'units'} in the smaller group - "
                        "results this small swing wildly." if small else "",
                        "Run the test longer before deciding." if small else ""))

    # 3. Significance, corrected for how many metrics were checked.
    p_adj = min(1.0, p * n_metrics)
    sig = p_adj < alpha
    checks.append(Check("Difference is real", PASS if sig else FAIL,
                        "" if sig else f"{variant} vs {control}: {lift:+.1%} lift, but p = {p_adj:.3f}"
                        f"{f' after correcting for {n_metrics} metrics' if n_metrics > 1 else ''}. "
                        "That's well within what random noise produces.",
                        "" if sig else "Treat this as no difference. Don't ship based on it.",
                        {"lift": lift, "p": p, "p_adjusted": p_adj}))

    # 4. Implausible lift.
    if abs(lift) > 0.5 and np.isfinite(lift):
        checks.append(Check("Believable effect size", WARN,
                            f"A {lift:+.0%} lift is far bigger than almost any real product change produces.",
                            "Check for a tracking bug or a difference in who landed in each group."))

    # 5. Outlier dependence (numeric outcomes).
    if not binary and sig:
        cap = np.quantile(np.concatenate([a, b]), 0.99)
        _, p_trim = _compare(np.minimum(a, cap), np.minimum(b, cap), False)
        bad = p_trim * n_metrics >= alpha
        checks.append(Check("Not driven by a few big values", FAIL if bad else PASS,
                            "Cap the top 1% of values and the difference disappears - a handful of whales "
                            "made this result." if bad else "",
                            "Report the capped result, or run longer until it holds without the outliers."
                            if bad else ""))

    # 6. Peeking + novelty, if we know when each row happened.
    if time_col:
        d = df.sort_values(time_col)
        is_b = (d[group_col].astype(str) == variant).to_numpy()
        y = d[outcome_col].to_numpy(float)
        cuts = np.linspace(0.2, 1.0, 9)
        ps = []
        for c in cuts:
            k = int(len(d) * c)
            ya, yb = y[:k][~is_b[:k]], y[:k][is_b[:k]]
            ps.append(_compare(ya, yb, binary)[1] if len(ya) > 1 and len(yb) > 1 else 1.0)
        peeked = sig and all(q >= alpha for q in ps[:-1])
        checks.append(Check("Not stopped at a lucky moment", WARN if peeked else PASS,
                            "It was never significant at any earlier point - it only crossed the line at the "
                            "very end. If you stopped the test the moment it looked good, that's peeking." if peeked else "",
                            "Decide the sample size up front, or use a sequential test built for peeking." if peeked else "",
                            {"p_over_time": dict(zip([f"{c:.0%}" for c in cuts], ps))}))
        half = len(d) // 2
        l1, _ = _compare(y[:half][~is_b[:half]], y[:half][is_b[:half]], binary)
        l2, _ = _compare(y[half:][~is_b[half:]], y[half:][is_b[half:]], binary)
        fading = l1 > 0 and l2 <= 0
        checks.append(Check("Effect lasts", WARN if fading else PASS,
                            f"Lift was {l1:+.1%} in the first half and {l2:+.1%} in the second - looks like a "
                            "novelty bump that wore off." if fading else "",
                            "Run longer and judge on the later period." if fading else "",
                            {"lift_first_half": l1, "lift_second_half": l2}))
    else:
        checks.append(Check("Not stopped at a lucky moment", SKIP, "Add a timestamp column to check for peeking."))

    return report("A/B test / experiment", checks, n=n, unit="units")
