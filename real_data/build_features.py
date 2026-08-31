#!/usr/bin/env python3
"""
Turn raw Rithmic ES ticks into a small, generic, CAUSAL feature set + aligned
forward returns - the two arrays core.generator.Generator.generate() needs.

Deliberately simple and general (volume delta, realized vol, autocorrelation,
tick intensity, range) - NOT the other STARK's proprietary footprint/absorption
logic, per VISION/CLAUDE.md's isolation rule. This is a first honest look at
real data, not a feature-engineering project.

Every feature at bar i is computed ONLY from ticks up to and including bar i.
Forward return is bar (i+1)'s close vs bar i's close - one bar ahead, no
lookahead.
"""
import json
import sys
from pathlib import Path

import numpy as np

TICKS_FILE = Path(__file__).parent / "es_ticks_raw.jsonl"
OUT_FILE = Path(__file__).parent / "es_features.json"

TICKS_PER_BAR = 50  # ticks per synthetic bar - tick-count bars, not time bars


def load_ticks() -> list[dict]:
    ticks = []
    with open(TICKS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                ticks.append(json.loads(line))
    return ticks


def find_field(tick: dict, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in tick:
            return c
    return None


def main():
    ticks = load_ticks()
    if not ticks:
        print(f"No ticks in {TICKS_FILE}. Run pull_ticks.py first.")
        sys.exit(1)

    print(f"Loaded {len(ticks)} raw ticks.")
    print("Fields present on tick 0:", sorted(ticks[0].keys()))

    price_field = find_field(ticks[0], ["price", "trade_price", "last_trade_price", "close_price"])
    size_field = find_field(ticks[0], ["size", "volume", "trade_size"])

    if price_field is None:
        print("FATAL: could not find a price field in the raw tick data. Aborting - "
              "not fabricating a feature set from data we can't read.")
        sys.exit(1)

    print(f"Using price field: {price_field!r}")
    print(f"Using size field: {size_field!r}" if size_field else "No size/volume field found - volume features skipped.")

    prices = np.array([float(t[price_field]) for t in ticks])
    if size_field:
        size_missing = sum(1 for t in ticks if size_field not in t)
        if size_missing:
            print(f"WARNING: {size_missing} of {len(ticks)} ticks are missing "
                  f"'{size_field}' - using NaN for those (not fabricating a volume).")
        sizes = np.array([float(t[size_field]) if size_field in t else np.nan for t in ticks])
    else:
        sizes = np.ones(len(ticks))

    # Real aggressor-side signal, confirmed present in this data: each 1-trade
    # bar carries EITHER ask_volume (buy-initiated, hit the offer) OR
    # bid_volume (sell-initiated, hit the bid) - never both, never a generic
    # "side" field. ~52%/48% split across a 50k sample, so this is genuine
    # order flow, not a fabricated proxy.
    has_side = any(("ask_volume" in t or "bid_volume" in t) for t in ticks)
    if has_side:
        sides = np.array([
            1.0 if "ask_volume" in t else (-1.0 if "bid_volume" in t else 0.0)
            for t in ticks
        ])
        print("Using real aggressor side: presence of ask_volume (+1, buy) vs bid_volume (-1, sell).")
    else:
        sides = None
        print("No ask_volume/bid_volume fields found - volume-delta feature skipped (labelled honestly, not faked).")

    n_bars = len(prices) // TICKS_PER_BAR
    if n_bars < 40:
        print(f"FATAL: only {n_bars} bars of {TICKS_PER_BAR} ticks each - too few for any "
              f"honest split. Pull more history or shrink TICKS_PER_BAR.")
        sys.exit(1)

    bar_close = np.array([prices[(i + 1) * TICKS_PER_BAR - 1] for i in range(n_bars)])
    bar_high = np.array([prices[i * TICKS_PER_BAR:(i + 1) * TICKS_PER_BAR].max() for i in range(n_bars)])
    bar_low = np.array([prices[i * TICKS_PER_BAR:(i + 1) * TICKS_PER_BAR].min() for i in range(n_bars)])
    bar_volume = np.array([sizes[i * TICKS_PER_BAR:(i + 1) * TICKS_PER_BAR].sum() for i in range(n_bars)])

    bar_return = np.diff(bar_close, prepend=bar_close[0]) / np.where(bar_close == 0, 1, bar_close)
    bar_return[0] = 0.0

    features: dict[str, list[float]] = {}

    # 1. Prior-bar return (momentum / mean-reversion candidate).
    features["ret_lag1"] = list(bar_return)

    # 2. Realized vol within the bar (tick-level return std).
    realized_vol = []
    for i in range(n_bars):
        seg = prices[i * TICKS_PER_BAR:(i + 1) * TICKS_PER_BAR]
        rets = np.diff(seg) / np.where(seg[:-1] == 0, 1, seg[:-1])
        realized_vol.append(float(np.std(rets)) if len(rets) else 0.0)
    features["realized_vol"] = realized_vol

    # 3. Bar range (high-low), normalized by close.
    features["bar_range_pct"] = list((bar_high - bar_low) / np.where(bar_close == 0, 1, bar_close))

    # 4. Volume relative to its own trailing average (20-bar), a crude "activity" feature.
    vol_avg = np.array([bar_volume[max(0, i - 20):i + 1].mean() for i in range(n_bars)])
    features["volume_vs_trailing_avg"] = list(bar_volume / np.where(vol_avg == 0, 1, vol_avg))

    # 5. Two-bar return autocorrelation proxy: ret_lag1 * ret_lag2 sign product.
    ret_lag2 = np.roll(bar_return, 1)
    ret_lag2[0] = 0.0
    features["return_persistence"] = list(np.sign(bar_return) * np.sign(ret_lag2))

    # 6. Volume delta, only if we actually have a real side field - not fabricated.
    if sides is not None:
        vol_delta = []
        for i in range(n_bars):
            seg_sides = sides[i * TICKS_PER_BAR:(i + 1) * TICKS_PER_BAR]
            seg_sizes = sizes[i * TICKS_PER_BAR:(i + 1) * TICKS_PER_BAR]
            vol_delta.append(float(np.sum(seg_sides * seg_sizes)))
        features["volume_delta"] = vol_delta
    else:
        print("volume_delta OMITTED - no real aggressor-side field in the raw ticks.")

    # Forward return: next bar's return, aligned so features[i] predicts fwd[i].
    forward_returns = list(np.concatenate([bar_return[1:], [np.nan]]))

    # Drop the last row (no forward return) and the first couple (warm-up lags).
    valid = slice(2, n_bars - 1)
    features = {k: v[valid] for k, v in features.items()}
    forward_returns = forward_returns[valid]

    n = len(forward_returns)
    print(f"\nBuilt {len(features)} features over {n} bars ({TICKS_PER_BAR} ticks/bar).")
    for k in features:
        print(f"  {k}: n={len(features[k])}")

    with open(OUT_FILE, "w") as f:
        json.dump({"features": features, "forward_returns": forward_returns,
                   "n_bars": n, "ticks_per_bar": TICKS_PER_BAR,
                   "n_raw_ticks": len(ticks)}, f)
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
