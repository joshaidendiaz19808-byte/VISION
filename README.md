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

## STARK MK18 (`stark-core/`)
VISION judges a finished strategy. STARK is the engine that proposes them and
decides what they're allowed to risk. Four levels, each one only ever able to
*reduce* conviction:

| Level | Module | Job |
|---|---|---|
| 1 | `core/generator.py` | proposes falsifiable claims, shrunk by sample size |
| 1 | `core/red_team.py` | six adversarial attacks; survivors only |
| 1 | `core/risk_arbiter.py` | hard symbolic guardrails — the veto is final |
| 2 | `simulation/causal_engine.py` | do() interventions + counterfactual branches |
| 2 | `simulation/active_inference.py` | surprise → belief update → what to test next |
| 3 | `tensors/market_tensors.py` | temperature / entropy / energy / pressure fields |
| 4 | `interface/jarvis_node.py` | narrates the whole chain in plain English |

Read `stark-core/CLAUDE.md` first — it holds the architecture rules.
Every module runs standalone (`python3 stark-core/core/red_team.py`).
Tests: `python3 stark-core/tests/test_smoke.py`
