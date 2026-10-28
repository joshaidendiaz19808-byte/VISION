# VISION — Is your result real?
The bullshit-detector for results — trading, experiments, ML/AI models, research data, and the code behind them.

Input: a results file, an API connection (Alpaca, any CSV/JSON URL with your key), and/or the code (files, notebooks, zip, GitHub repo)
Output: LOOKS REAL / QUESTIONABLE / LOOKS FAKE + WHY + how to fix it

Run: `streamlit run app.py` (one page). Tests: `pytest tests/test_checks.py` — every check is tested on planted fakes and planted clean cases.

## What it checks (`checks/`)
| Module | Checks |
|---|---|
| `trading.py` | sample size, out-of-sample decay, remove-best-trades, cost stress, Monte Carlo luck, deflated Sharpe (how many versions you tried), smoothed returns |
| `ab_test.py` | sample-ratio mismatch, significance with multiple-metric correction, enough data, implausible lift, whale-driven results, peeking, novelty decay |
| `ml_model.py` | train/test duplicate rows, entity overlap, train-on-future, single-column target leakage, score vs dumb baseline with bootstrap CI, suspiciously perfect scores |
| `integrity.py` | Benford first digits, last-digit preference, duplicate padding, p-hacking bump under 0.05, GRIM (impossible means) |
| `code_audit.py` | negative shift, next-bar peeking, centered windows, bfill, whole-history normalization, preprocessing fit before split, shuffled time splits, best-version-on-test-set, seed fishing, scoring on train, survivorship lists, no costs, unseeded randomness, notebooks run out of order — static `ast` only, never executes code |
| `connectors.py` | GitHub repo (token for private), any URL (CSV/JSON + auth header), Alpaca portfolio history — keys live in memory only |

| `ai_review.py` | Claude (Opus 5.5) reviews the code as untrusted data, using static hits as leads — optional, needs an Anthropic key |

## Hosting
- Login turns on when `.streamlit/secrets.toml` (or the host's secrets) has an `[auth]` section — see `.streamlit/secrets.toml.example` (Google login).
- Hosted users bring their own Anthropic key for the AI review unless `allow_server_ai_key = true`.
- Deploy: Streamlit Community Cloud (point at this repo, `app.py`, paste secrets) or any Docker host (`Dockerfile`, honours `$PORT`).

## Known limitations
- Code audit is pattern-based: it catches the common ways results get faked, not every bug. Python + notebooks only.
- Forensic data checks (Benford, digits) are red flags for a human to look at, not proof.
- Alpaca, IBKR and MLflow connectors and the AI review are tested against canned responses — not yet against real accounts.
- Dockerfile not yet built (no Docker on the dev machine).

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
