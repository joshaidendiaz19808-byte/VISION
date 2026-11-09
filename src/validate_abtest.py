#!/usr/bin/env python3
"""Does the A/B checker actually work? Run it on experiments where we KNOW the truth."""
import numpy as np, pandas as pd
from abtest import check_counts, check_rows

rng = np.random.default_rng(7)
N = 400
ok = True

def rows(n, rate_a, rate_b, days=14, split=0.5):
    v = np.where(rng.random(n) < split, "B", "A")
    conv = (rng.random(n) < np.where(v == "B", rate_b, rate_a)).astype(int)
    d = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, days, n), unit="D")
    return pd.DataFrame({"variant": v, "converted": conv, "date": d})

# 1) No real difference (A/A): how often do we falsely call a winner?
fooled = sum(check_rows(rows(20000, 0.05, 0.05), "variant", "converted", "date")["is_real"] for _ in range(N))
# 2) Real +20% lift (5% -> 6%): how often do we catch it?
caught = sum(check_rows(rows(20000, 0.05, 0.06), "variant", "converted", "date")["is_real"] for _ in range(N))
print(f"Conversion tests ({N} each, 20k users):")
print(f"  Fooled by luck (no real difference, called a winner): {fooled}/{N} ({fooled/N*100:.1f}%)  <- ~5% by design (95% confidence)")
print(f"  Caught a real +20% lift:                              {caught}/{N} ({caught/N*100:.1f}%)")
# 95% confidence means ~5% false alarms by design; allow sampling wobble on 400 runs.
ok &= fooled / N <= 0.075 and caught / N > 0.8

# 3) Revenue per user, no real difference, heavy-tailed spend
def revenue(n, mult_b=1.0, whale=0.0):
    v = np.where(rng.random(n) < 0.5, "A", "B")
    spend = np.where(rng.random(n) < 0.1, rng.lognormal(3, 1, n), 0.0)
    spend = spend * np.where(v == "B", mult_b, 1.0)
    df = pd.DataFrame({"variant": v, "revenue": spend})
    if whale:
        df.loc[len(df)] = ["B", whale]
    return df
fooled_rev = sum(check_rows(revenue(10000), "variant", "revenue")["is_real"] for _ in range(N))
caught_rev = sum(check_rows(revenue(10000, 1.3), "variant", "revenue")["is_real"] for _ in range(N))
print(f"Revenue tests ({N} each, 10k users):")
print(f"  Fooled by luck: {fooled_rev}/{N} ({fooled_rev/N*100:.1f}%)  <- ~5% or less")
print(f"  Caught a real +30% revenue lift: {caught_rev}/{N} ({caught_rev/N*100:.1f}%)")
ok &= fooled_rev / N <= 0.075 and caught_rev / N > 0.5

# 4) Targeted traps - each must be flagged NOT PROVEN
traps = {
    "one whale customer fakes a revenue win": check_rows(revenue(10000, whale=250000), "variant", "revenue"),
    "broken 55/45 traffic split": check_rows(rows(20000, 0.05, 0.06, split=0.55), "variant", "converted", "date"),
    "tiny test (200 visitors)": check_counts(100, 3, 100, 9),
}
# novelty: B wins big in week 1, loses in week 2
early = rows(20000, 0.05, 0.08, days=7)
late = rows(20000, 0.05, 0.04, days=7); late["date"] += pd.Timedelta(days=7)
traps["novelty effect that fades"] = check_rows(pd.concat([early, late]), "variant", "converted", "date")
print("Trap cases (each must be caught):")
for name, out in traps.items():
    flagged = not out["is_real"]
    ok &= flagged
    print(f"  {'CAUGHT' if flagged else 'MISSED'}  {name}")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
