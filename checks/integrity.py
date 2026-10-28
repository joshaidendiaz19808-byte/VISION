"""Any numbers from any field: signs the data itself was fabricated, massaged or p-hacked.

These are forensic red flags, not proof - a WARN means "a human should look",
which is how auditors and research-integrity teams use them.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .common import Check, report, PASS, WARN, FAIL, SKIP

BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def benford(name: str, x: pd.Series) -> Check:
    """Natural quantities spanning several orders of magnitude follow Benford's first-digit law."""
    v = x[x > 0].astype(float)
    title = f"'{name}' first digits look natural"
    if len(v) < 300 or np.log10(v.max() / v.min()) < 2:
        return Check(title, SKIP, "Benford only applies to 300+ values spanning 100x or more.")
    first = (v / 10 ** np.floor(np.log10(v))).astype(int).clip(1, 9)
    obs = np.bincount(first, minlength=10)[1:] / len(first)
    mad = float(np.abs(obs - BENFORD).mean())  # Nigrini: > 0.015 = nonconformity
    bad = mad > 0.015
    return Check(title, WARN if bad else PASS,
                 f"Leading digits of '{name}' don't follow the pattern natural data does (deviation {mad:.3f}, "
                 "limit 0.015). Invented numbers usually fail this." if bad else "",
                 "Trace a sample of these values back to their source records." if bad else "",
                 {"mad": mad})


def last_digit(name: str, raw: pd.Series) -> Check:
    """When people make numbers up, the last digit isn't uniform (too many 0s and 5s, too few 7s...)."""
    title = f"'{name}' last digits look natural"
    s = raw.dropna().astype(str).str.strip().str.replace(r"[^0-9.]", "", regex=True)
    digits = s.str.replace(".", "", regex=False).str.lstrip("0")
    digits = digits[digits.str.len() >= 3].str[-1].astype(int)
    if len(digits) < 200:
        return Check(title, SKIP, "Needs 200+ values with 3+ significant digits.")
    counts = np.bincount(digits, minlength=10)
    p = stats.chisquare(counts).pvalue
    bad = p < 0.001
    top = int(np.argmax(counts))
    return Check(title, WARN if bad else PASS,
                 f"The last digit of '{name}' is far from random (p = {p:.1e}; '{top}' shows up "
                 f"{counts[top] / counts.sum():.0%} of the time vs 10% expected). That's a sign of rounding by hand "
                 "or made-up values." if bad else "",
                 "Check how these were recorded - instrument, export or typed in?" if bad else "",
                 {"p": float(p)})


def duplicates(df: pd.DataFrame) -> Check:
    rate = float(df.duplicated().mean())
    bad = rate > 0.05
    return Check("Not padded with copies", WARN if bad else PASS,
                 f"{rate:.0%} of rows are exact copies of another row. Duplicates inflate sample size and "
                 "make results look more certain than they are." if bad else "",
                 "Remove duplicates and re-run your analysis." if bad else "", {"dup_rate": rate})


def p_hacking(p: pd.Series) -> Check:
    """p-values that bunch up just under 0.05 are the fingerprint of tweaking until it's 'significant'."""
    p = p[(p > 0) & (p < 1)]
    just_under = int(((p > 0.045) & (p <= 0.05)).sum())
    below = int(((p > 0.04) & (p <= 0.045)).sum())
    title = "No p-hacking bump just under 0.05"
    if just_under + below < 10:
        return Check(title, SKIP, "Need 10+ p-values between 0.04 and 0.05 to check.")
    # Real effects make p-values *decrease* in density toward 0.05; a bump is suspicious.
    pv = stats.binomtest(just_under, just_under + below, 0.5, alternative="greater").pvalue
    bad = pv < 0.05
    return Check(title, FAIL if bad else PASS,
                 f"{just_under} p-values sit in 0.045-0.05 vs {below} in 0.04-0.045. Real findings don't pile "
                 "up right at the significance line - this looks like analyses were tweaked until they passed."
                 if bad else "",
                 "Pre-register the analysis and report every test you ran, not just the ones that worked."
                 if bad else "", {"bump_p": float(pv)})


def grim(df: pd.DataFrame, mean_col: str, n_col: str, items_col: str | None = None) -> Check:
    """GRIM test: a mean of whole-number answers from n people can only take certain values."""
    bad_rows = []
    for i, row in df[[c for c in (mean_col, n_col, items_col) if c]].dropna().iterrows():
        raw = str(row[mean_col])
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        n = int(row[n_col]) * (int(row[items_col]) if items_col else 1)
        m = float(row[mean_col])
        if decimals == 0 or n <= 0 or n >= 10 ** decimals:
            continue
        possible = any(round(k / n, decimals) == round(m, decimals)
                       for k in (np.floor(m * n), np.ceil(m * n)))
        if not possible:
            bad_rows.append(i)
    return Check("Reported means are mathematically possible", FAIL if bad_rows else PASS,
                 f"{len(bad_rows)} reported mean(s) can't come from whole-number answers with that many people "
                 f"(rows {bad_rows[:10]}). Either a typo or the numbers weren't computed from real data."
                 if bad_rows else "",
                 "Re-derive those means from the raw responses." if bad_rows else "")


def run(df: pd.DataFrame, raw: pd.DataFrame | None = None) -> dict:
    """df: parsed table. raw: the same table read as strings (keeps the digits people typed)."""
    raw = raw if raw is not None else df.astype(str)
    checks = [duplicates(df)]
    lower = {c.lower().replace("_", "").replace("-", ""): c for c in df.columns}
    pcol = next((lower[k] for k in ("p", "pvalue", "pval") if k in lower), None)
    if pcol:
        checks.append(p_hacking(pd.to_numeric(df[pcol], errors="coerce").dropna()))
    if "mean" in lower and "n" in lower:
        checks.append(grim(raw, lower["mean"], lower["n"], lower.get("items")))
    for c in df.columns:
        if c == pcol:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().mean() < 0.9 or x.nunique() < 20:
            continue
        if c.lower() == "id" or c.lower().endswith(("_id", "id")) and x.is_unique:
            continue  # identifiers aren't measurements
        checks.append(benford(c, x.dropna()))
        checks.append(last_digit(c, raw[c]))
    return report("Data integrity", checks, n=len(df), unit="rows")
