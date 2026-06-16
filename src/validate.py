#!/usr/bin/env python3
"""Does VISION actually work? Run it on data where we KNOW the truth."""
import pandas as pd, numpy as np
from vision import verdict

def is_real(df): return "REAL" in verdict(df)["verdict"]

rng = np.random.default_rng(123)
N = 300
fooled, missed = 0, 0

for _ in range(N):
    noise = pd.DataFrame({"date": range(200), "return_pct": rng.normal(0, 0.8, 200)})
    if is_real(noise): fooled += 1
    real = pd.DataFrame({"date": range(200), "return_pct": rng.normal(0.25, 0.8, 200)})
    if not is_real(real): missed += 1

print(f"Tested {N} fake strategies + {N} real ones.")
print(f"  Fooled by luck (called noise REAL): {fooled}/{N}  ({fooled/N*100:.1f}%)  <- must be ~0")
print(f"  Missed a real edge (called it FAKE): {missed}/{N}  ({missed/N*100:.1f}%)")
