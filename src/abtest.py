#!/usr/bin/env python3
"""
VISION - A/B test checker.

Same idea as the strategy checker: "is this win real, or just luck?"

Two ways in:
  check_counts(...)  - you only have totals: visitors + conversions per version
  check_rows(df,...) - one row per visitor/user: which version they saw + the result
                       (0/1 converted, OR a number like revenue / time on page)

Output has the same shape as vision.verdict(): verdict, score, reasons, fixes,
plus 'summary' numbers (rates, lift, p-value, confidence interval).

No scipy needed - normal / chi-square tails come from math.erfc.
"""
import math
import numpy as np
import pandas as pd

ALPHA = 0.05          # standard 95% confidence
SRM_ALPHA = 0.001     # traffic-split check is deliberately strict
MIN_PER_GROUP = 100   # conversions (yes/no metric) or users (number metric)

CONTROL_NAMES = {"a", "control", "original", "baseline", "old", "ctrl", "0", "v1", "current"}


# ── tiny stats helpers ─────────────────────────────────────────────────────────

def _p_two_sided(z):
    return math.erfc(abs(z) / math.sqrt(2))


def _srm_p(n_a, n_b):
    """Chi-square (1 df) test that traffic was split 50/50."""
    total = n_a + n_b
    exp = total / 2
    chi2 = (n_a - exp) ** 2 / exp + (n_b - exp) ** 2 / exp
    return math.erfc(math.sqrt(chi2 / 2))


def _prop_test(n_a, c_a, n_b, c_b):
    """Two-proportion z-test. Returns (diff, p, ci_low, ci_high) for rate_b - rate_a."""
    p_a, p_b = c_a / n_a, c_b / n_b
    pooled = (c_a + c_b) / (n_a + n_b)
    se_pool = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    diff = p_b - p_a
    p = 1.0 if se_pool == 0 else _p_two_sided(diff / se_pool)
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    return diff, p, diff - 1.96 * se, diff + 1.96 * se


def _mean_test(a, b):
    """Welch-style z-test on means (fine at the sample sizes we require)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = b.mean() - a.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if min(len(a), len(b)) > 1 else 0.0
    p = 1.0 if se == 0 else _p_two_sided(diff / se)
    return diff, p, diff - 1.96 * se, diff + 1.96 * se


def _test(passed, reason=None, fix=None, name=""):
    return {"name": name, "passed": bool(passed),
            "reason": None if passed else reason, "fix": None if passed else fix}


# ── the honesty tests ──────────────────────────────────────────────────────────

def test_sample_size(n_a, n_b, label_a, label_b, unit):
    small = min(n_a, n_b)
    return _test(
        small >= MIN_PER_GROUP,
        f"Not enough data. The smaller group has only {small:,} {unit}; you need {MIN_PER_GROUP}+ in each. "
        "With less than that, a 'winner' is often just noise.",
        "Keep the test running until both versions have 100+ "
        f"{unit} each. Decide the end date BEFORE you start, and don't stop early because one side is ahead.",
        "Enough data",
    )


def test_traffic_split(n_a, n_b, label_a, label_b):
    p = _srm_p(n_a, n_b)
    share = n_a / (n_a + n_b) * 100
    return _test(
        p >= SRM_ALPHA,
        f"Your traffic split is broken. You meant to split 50/50, but {label_a} got {share:.1f}% and "
        f"{label_b} got {100 - share:.1f}%. That's too lopsided to be chance (p={p:.1g}), so something in "
        "the setup is filtering who sees which version - and that can fake a win.",
        "Find the bug before trusting any result: redirects, bots, caching, or one version loading slower "
        "and losing visitors. (If you split traffic unevenly on purpose, ignore this.)",
        "Fair traffic split",
    )


def test_significance(diff, p, label_a, label_b, fmt):
    winner, loser = (label_b, label_a) if diff > 0 else (label_a, label_b)
    return _test(
        p < ALPHA,
        f"The difference could easily be luck. {winner} is ahead by {fmt(abs(diff))}, but there's a "
        f"{p * 100:.0f}% chance of a gap at least that big even if both versions were identical. "
        "You need under 5%.",
        "Treat this as 'no proven difference'. Run longer for more data, or test a bolder change - "
        "small tweaks need huge samples to prove anything.",
        "Beats luck",
    )


def test_holds_over_time(halves, label_a, label_b, overall_diff):
    """halves: list of (diff_first_half, diff_second_half). Pass if both halves agree with the overall winner."""
    d1, d2 = halves
    sign = np.sign(overall_diff)
    passed = sign != 0 and np.sign(d1) == sign and np.sign(d2) == sign
    winner = label_b if overall_diff > 0 else label_a
    return _test(
        passed,
        f"The win doesn't hold up over time. {winner} won overall, but not in both the first and the second "
        "half of the test. That's a classic sign of a novelty bump or one weird day doing all the work.",
        "Look at the results day by day. If the lead faded, run it longer and see if it comes back before "
        "you roll it out.",
        "Holds up over time",
    )


def test_outliers(a, b, label_a, label_b, overall_diff, top_frac=0.001, min_drop=5):
    """Drop the few biggest values (top 0.1%, at least 5 rows) across both groups.
    The same version should still win significantly - otherwise a handful of 'whales' made the result."""
    k = max(min_drop, int(np.ceil((len(a) + len(b)) * top_frac)))
    allv = np.concatenate([a, b])
    is_b = np.r_[np.zeros(len(a), bool), np.ones(len(b), bool)]
    keep = np.ones(len(allv), bool)
    keep[np.argsort(allv)[::-1][:k]] = False
    a2, b2 = allv[keep & ~is_b], allv[keep & is_b]
    winner = label_b if overall_diff > 0 else label_a
    reason = (f"The win rides on a few huge values. Remove just the {k} biggest rows (e.g. a couple of big "
              f"spenders) and {winner} no longer clearly wins.")
    fix = ("One whale customer can flip a revenue test. Check those top rows - are they real, repeatable "
           "customers, or a one-off / a data error?")
    if min(len(a2), len(b2)) < 2:
        return _test(False, reason, fix, "Not just outliers")
    d2, p2, _, _ = _mean_test(a2, b2)
    passed = np.sign(d2) == np.sign(overall_diff) and p2 < ALPHA
    return _test(passed, reason, fix, "Not just outliers")


# ── putting it together ────────────────────────────────────────────────────────

def _finish(tests, summary, label_a, label_b):
    passed = sum(t["passed"] for t in tests)
    total = len(tests)
    diff = summary["diff"]
    if passed == total:
        winner = label_b if diff > 0 else label_a
        verdict = f"REAL WIN - {winner} is better"
        is_real = True
    else:
        verdict = "NOT PROVEN - don't roll this out yet"
        is_real = False
    return {
        "verdict": verdict,
        "is_real": is_real,
        "score": f"{passed}/{total} tests passed",
        "tests": tests,
        "reasons": [t["reason"] for t in tests if not t["passed"]],
        "fixes": [t["fix"] for t in tests if not t["passed"]],
        "summary": summary,
    }


def _pct(x):
    return f"{x * 100:.2f} points"


def check_counts(visitors_a, conversions_a, visitors_b, conversions_b, label_a="A", label_b="B"):
    """Totals only (no per-day data, so the 'holds up over time' test is skipped)."""
    n_a, c_a, n_b, c_b = int(visitors_a), int(conversions_a), int(visitors_b), int(conversions_b)
    if min(n_a, n_b) <= 0:
        raise ValueError("Each version needs at least 1 visitor.")
    if c_a < 0 or c_b < 0 or c_a > n_a or c_b > n_b:
        raise ValueError("Conversions must be between 0 and the number of visitors.")
    diff, p, lo, hi = _prop_test(n_a, c_a, n_b, c_b)
    rate_a, rate_b = c_a / n_a, c_b / n_b
    tests = [
        test_sample_size(min(c_a, n_a - c_a), min(c_b, n_b - c_b), label_a, label_b, "conversions"),
        test_traffic_split(n_a, n_b, label_a, label_b),
        test_significance(diff, p, label_a, label_b, _pct),
    ]
    summary = {
        "metric": "conversion rate", "kind": "binary",
        "label_a": label_a, "label_b": label_b, "n_a": n_a, "n_b": n_b,
        "value_a": rate_a, "value_b": rate_b, "diff": diff,
        "lift": (rate_b / rate_a - 1) if rate_a > 0 else float("nan"),
        "p_value": p, "ci_low": lo, "ci_high": hi,
    }
    return _finish(tests, summary, label_a, label_b)


_TRUE = {"1", "true", "yes", "y", "t", "converted", "purchase", "purchased"}
_FALSE = {"0", "false", "no", "n", "f", "", "nan", "none"}


def _to_metric(series):
    """Return (numeric array, kind). Yes/no style columns become 0/1."""
    s = series.dropna()
    as_str = s.astype(str).str.strip().str.lower()
    if len(as_str) and as_str.isin(_TRUE | _FALSE).all():
        return as_str.isin(_TRUE).astype(float), "binary"
    num = pd.to_numeric(s, errors="coerce")
    num = num[np.isfinite(num)]
    uniq = set(num.unique())
    kind = "binary" if uniq <= {0.0, 1.0} else "numeric"
    return num.astype(float), kind


def pick_control(labels):
    """Guess which variant is the original. Falls back to the first label."""
    for lab in labels:
        if str(lab).strip().lower() in CONTROL_NAMES:
            return lab
    return labels[0]


def check_rows(df, variant_col, metric_col, date_col=None, control=None):
    """One row per user. variant_col must have exactly 2 values."""
    work = df[[variant_col, metric_col] + ([date_col] if date_col else [])].copy()
    work = work.dropna(subset=[variant_col]).reset_index(drop=True)
    labels = list(pd.unique(work[variant_col]))
    if len(labels) != 2:
        raise ValueError(
            f"The '{variant_col}' column should have exactly 2 versions (like A and B). "
            f"Found {len(labels)}: {labels[:6]}"
        )
    control = control if control in labels else pick_control(labels)
    treat = labels[1] if labels[0] == control else labels[0]
    label_a, label_b = str(control), str(treat)

    metric, kind = _to_metric(work[metric_col])
    work = work.loc[metric.index].assign(_m=metric)
    a = work.loc[work[variant_col] == control, "_m"].to_numpy()
    b = work.loc[work[variant_col] == treat, "_m"].to_numpy()
    if min(len(a), len(b)) == 0:
        raise ValueError("One of the versions has no usable numbers.")

    if kind == "binary":
        diff, p, lo, hi = _prop_test(len(a), a.sum(), len(b), b.sum())
        size_a, size_b, unit, fmt = min(a.sum(), len(a) - a.sum()), min(b.sum(), len(b) - b.sum()), "conversions", _pct
        metric_name = "conversion rate"
    else:
        diff, p, lo, hi = _mean_test(a, b)
        size_a, size_b, unit = len(a), len(b), "users"
        fmt = lambda x: f"{x:,.2f} per user"
        metric_name = f"average {metric_col}"

    tests = [
        test_sample_size(int(size_a), int(size_b), label_a, label_b, unit),
        test_traffic_split(len(a), len(b), label_a, label_b),
        test_significance(diff, p, label_a, label_b, fmt),
    ]

    if date_col:
        dates = pd.to_datetime(work[date_col], errors="coerce")
        if dates.notna().mean() > 0.9 and dates.nunique() > 1:
            mid = dates.quantile(0.5)
            halves = []
            for mask in (dates <= mid, dates > mid):
                h = work[mask]
                ha = h.loc[h[variant_col] == control, "_m"]
                hb = h.loc[h[variant_col] == treat, "_m"]
                halves.append((hb.mean() - ha.mean()) if len(ha) and len(hb) else 0.0)
            tests.append(test_holds_over_time(halves, label_a, label_b, diff))

    if kind == "numeric":
        tests.append(test_outliers(a, b, label_a, label_b, diff))

    va, vb = a.mean(), b.mean()
    summary = {
        "metric": metric_name, "kind": kind,
        "label_a": label_a, "label_b": label_b, "n_a": len(a), "n_b": len(b),
        "value_a": va, "value_b": vb, "diff": diff,
        "lift": (vb / va - 1) if va != 0 else float("nan"),
        "p_value": p, "ci_low": lo, "ci_high": hi,
    }
    return _finish(tests, summary, label_a, label_b)


def print_report(out):
    s = out["summary"]
    line = "=" * 56
    print(f"\n{line}\n VISION - A/B test check\n{line}")
    fmt = (lambda v: f"{v * 100:.2f}%") if s["kind"] == "binary" else (lambda v: f"{v:,.2f}")
    print(f" {s['label_a']}: {fmt(s['value_a'])} ({s['n_a']:,} users)")
    print(f" {s['label_b']}: {fmt(s['value_b'])} ({s['n_b']:,} users)")
    if not math.isnan(s["lift"]):
        print(f" Lift: {s['lift'] * 100:+.1f}%   p-value: {s['p_value']:.3g}\n")
    print(f" VERDICT: {out['verdict']}")
    print(f" SCORE:   {out['score']}\n")
    for r, f in zip(out["reasons"], out["fixes"]):
        print(f"  - {r}\n      fix: {f}")
    print(line)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) == 4 and all(x.isdigit() for x in args):
        print_report(check_counts(*map(int, args)))
    elif len(args) in (3, 4):
        df = pd.read_csv(args[0])
        print_report(check_rows(df, args[1], args[2], args[3] if len(args) == 4 else None))
    else:
        print("Usage:\n  python3 src/abtest.py VISITORS_A CONVERSIONS_A VISITORS_B CONVERSIONS_B\n"
              "  python3 src/abtest.py results.csv VARIANT_COLUMN METRIC_COLUMN [DATE_COLUMN]")
