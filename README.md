# VISION — Is it real, or just luck?
The bullshit-detector for results. Two checkers so far:
- **Trading strategies** (`src/vision.py`)
- **A/B tests** (`src/abtest.py`) — website/app experiments, conversion or revenue

Run the app: `pip install -r requirements.txt && streamlit run app.py`

## A/B test honesty tests
1. Enough data         — 100+ conversions (or users, for revenue) in each version
2. Fair traffic split  — catches broken 50/50 splits (sample ratio mismatch), which can fake a win
3. Beats luck          — standard 95% confidence (two-proportion / Welch z-test)
4. Holds up over time  — same winner in the first and second half (needs a date column; catches novelty bumps)
5. Not just outliers   — drop the biggest 0.1% of values; same winner (revenue tests only; catches "whales")

Command line:
- `python3 src/abtest.py 10000 500 10000 600` (visitors A, conversions A, visitors B, conversions B)
- `python3 src/abtest.py results.csv variant converted date`

Proof it works: `cd src && python3 validate_abtest.py` runs hundreds of fake experiments where the
truth is known — ~5% false alarms on no-difference tests (what 95% confidence means), catches a
real +20% conversion lift ~87% of the time, and catches all four trap cases (whale, broken split,
tiny sample, fading novelty).

Known limitation: "peeking" (checking every day and stopping as soon as it looks good) can't be
detected from the final data. Decide the test length before starting.

## Trading strategy checker

Input: a strategy's trade results (or rules + price data)
Output: REAL or FAKE + WHY + how to fix it

## The universal honesty tests (instrument-agnostic):
1. Out-of-sample decay  — does edge survive on unseen data?
2. Cost stress          — does it survive 2/5/10 bps?
3. Remove-best-trades   — is edge broad or just a few lucky wins?
4. Monte Carlo          — is the equity curve luck or skill?

## Known limitations:
- **Look-ahead bias cannot be detected from a trade log alone.**
  A trade log only records what happened, not how the strategy made its decisions.
  Catching look-ahead requires auditing the strategy's source code or backtesting engine directly.

## Why instrument-agnostic:
Overfitting looks the SAME in SPY, EUR/USD, gold, crypto.
The user brings their data + strategy. We bring the truth.

## Build log:
Day 1: __
