# VISION — Strategy Validation Engine
The bullshit-detector for trading strategies.

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
