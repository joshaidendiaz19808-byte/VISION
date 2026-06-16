#!/usr/bin/env python3
"""
VISION v1 - strategy bullshit detector.
Input: trades with columns [date, return_pct]  (0.5 == a 0.5% trade)
Output: REAL/FAKE verdict + plain-English reasons + how to fix.
"""
import pandas as pd
import numpy as np


def test_sample_size(returns, minimum=100):
    n = len(returns)
    passed = n >= minimum
    return {
        "passed": passed,
        "reason": None if passed else f"Not enough trades. You have {n}; you need 100+. Below that, a winning run is probably just luck.",
        "fix": None if passed else "Keep running the strategy (or test it on more markets) until you've got 100+ trades, then check again.",
    }


def test_out_of_sample(returns, split=0.7):
    n = len(returns)
    k = int(n * split)
    early_mean = returns.iloc[:k].mean()
    recent = returns.iloc[k:]
    recent_mean = recent.mean()
    passed = recent_mean > 0
    return {
        "passed": passed,
        "reason": None if passed else f"Your edge is fading. Your earlier trades averaged {early_mean:+.2f}% each, but your most recent {len(recent)} trades averaged {recent_mean:+.2f}% - it looks like it stopped working.",
        "fix": None if passed else "An edge that worked and then quit usually means the market caught on, or it was only ever tuned to old conditions. Don't trade it live until your recent trades are making money again.",
    }


def test_remove_best(returns):
    cut = returns.quantile(0.95)
    passed = returns[returns < cut].sum() > 0
    return {
        "passed": passed,
        "reason": None if passed else "Your profit is riding on a few lucky big wins. Drop your top 5% of trades and the rest lose money.",
        "fix": None if passed else "A real edge wins across lots of trades, not 2-3 monsters. Look at your biggest wins: can you actually repeat them, or did you just catch a one-off spike?",
    }


def test_cost_stress(returns, cost_bps=5.0):
    cost_pct = cost_bps / 100.0  # 5 bps = 0.05%
    passed = (returns - cost_pct).sum() > 0
    return {
        "passed": passed,
        "reason": None if passed else f"Wins on paper, loses for real. After fees + slippage ({cost_bps:.0f} bps a trade), it's a money-loser.",
        "fix": None if passed else "Your edge is smaller than your trading costs. Trade less often, aim for bigger moves, or use limit orders so you pay less getting in and out.",
    }


def test_monte_carlo(returns, n_sims=10000, seed=42):
    rng = np.random.default_rng(seed)
    arr = returns.to_numpy(dtype=float)
    n = len(arr)
    sims = arr[rng.integers(0, n, size=(n_sims, n))].sum(axis=1)
    p_luck = float((sims <= 0).mean())
    passed = p_luck < 0.01
    return {
        "passed": passed,
        "p_luck": p_luck,
        "reason": None if passed else f"This might just be luck. We remixed your own trades 10,000 times and {p_luck*100:.0f}% of those versions lost money. A real edge almost never does that.",
        "fix": None if passed else "Right now your results look like a coin flip that landed your way. Get more trades or a stronger signal before risking real cash.",
    }


def verdict(trades_df, cost_bps=5.0):
    df = trades_df.sort_values("date")
    returns = df["return_pct"].astype(float).reset_index(drop=True)
    tests = [
        test_sample_size(returns),
        test_out_of_sample(returns),
        test_remove_best(returns),
        test_cost_stress(returns, cost_bps=cost_bps),
        test_monte_carlo(returns),
    ]
    passed = sum(t["passed"] for t in tests)
    total = len(tests)
    return {
        "verdict": "LOOKS REAL - worth paper trading" if passed == total else "LOOKS FAKE / FRAGILE - don't trade this yet",
        "score": f"{passed}/{total} tests passed",
        "reasons": [t["reason"] for t in tests if not t["passed"]],
        "fixes": [t["fix"] for t in tests if not t["passed"]],
    }


def load_trades(path):
    """Read a real trades CSV. Needs a 'return_pct' column (0.5 == 0.5% gain)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "return_pct" not in df.columns:
        raise SystemExit(
            "Your CSV needs a column called 'return_pct' - the % return of each trade "
            "(0.5 = a 0.5% gain, -1.2 = a 1.2% loss).\n"
            f"What I found instead: {list(df.columns)}"
        )
    if "date" not in df.columns:
        df["date"] = range(len(df))  # no dates? use the row order
    return df[["date", "return_pct"]].dropna()


def print_report(path, df):
    out = verdict(df)
    line = "=" * 52
    print(f"\n{line}\n VISION - strategy check: {path}\n{line}")
    print(f" {len(df)} trades analyzed\n")
    print(f" VERDICT: {out['verdict']}")
    print(f" SCORE:   {out['score']}\n")
    if out["reasons"]:
        print(" What's wrong:")
        for r, f in zip(out["reasons"], out["fixes"]):
            print(f"  - {r}")
            print(f"      fix: {f}")
    else:
        print(" Passed every honesty test. Paper trade it before risking real money.")
    print(line)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print_report(sys.argv[1], load_trades(sys.argv[1]))
    else:
        print("Usage: python3 src/vision.py your_trades.csv")
