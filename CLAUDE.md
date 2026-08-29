# Project STARK: Autonomous AGI Capital & Simulation Organism

## Core Architecture Rules
1. **Separation of Concerns:** Neural networks (LLMs) handle intuition, pattern matching, and hypothesis generation. They are *never* trusted with absolute execution logic.
2. **Deterministic Safety:** All risk boundaries, invariant checks, and vetoes must live in pure, deterministic Python/NumPy code (`risk_arbiter.py`). No LLM can override a hard mathematical constraint.
3. **Active Inference over Rewards:** Optimization loops must minimize prediction error and free energy rather than relying on standard reinforcement learning reward functions.

## Build Phases
- **Phase 1:** Build the 3-node adversarial core (Generator -> Red Team -> Risk Arbiter).
- **Phase 2:** Implement the Generative Causal Simulator and Active Inference loop.
- **Phase 3:** Scale up to Energy-Based Tensors and Quantum-Inspired Stochastic Branching.

---

## Where the rules are enforced

The three rules above are not documentation. Each one is pinned by a test in
`stark-core/tests/test_smoke.py`, and breaking it turns the suite red:

| Rule | Enforced by | Test |
|---|---|---|
| 1. Separation of concerns | `Generator.scorer` is a pluggable hook that only ever emits a *prior*; the arbiter accepts plain primitives, never model objects | `test_risk_arbiter_imports_nothing_from_stark`, `test_confidence_only_ever_falls` |
| 2. Deterministic safety | `risk_arbiter.py` imports nothing from STARK; `RiskVeto` is the only sanctioned exit; every seed is explicit | `test_veto_is_final`, `test_gross_cap_is_absolute`, `test_no_bare_global_rng` |
| 3. Active inference | `ActiveInferenceLoop` minimizes free energy (complexity − accuracy) and expected free energy (pragmatic + epistemic). There is no reward function anywhere in the tree | `test_belief_moves_toward_evidence_and_tightens`, `test_curiosity_changes_what_gets_chosen` |

Rule 2 has a corollary worth stating outright: **the veto is unappealable.**
`RiskVeto` must never be caught and downgraded — catch it to report it, then
stop. Simulated evidence sizes nothing. A sub-100-observation track record is an
anecdote. These limits live in exactly one file and nowhere else.

## Repo map

```
VISION/                     the strategy validator — is a finished strategy real or fake?
├── CLAUDE.md               this file: project-wide law
├── app.py, src/vision.py   the four honesty tests + Streamlit UI
└── stark-core/             the engine — proposes strategies and sizes them
    ├── CLAUDE.md           STARK's own architecture rules (layers, imports, conventions)
    ├── core/               Phase 1: generator → red_team → risk_arbiter
    ├── simulation/         Phase 2: causal_engine + active_inference
    ├── tensors/            Phase 3: market_tensors (energy fields)
    └── interface/          jarvis_node — narrates the chain in plain English
```

`stark-core/CLAUDE.md` is the detail layer: dependency direction, the
standalone-module convention, and the honesty rules inherited from VISION. Read
it before editing anything under `stark-core/`. Where the two overlap, this file
wins.

## Phase status (MK18)

- **Phase 1 — built and hardened.** All three nodes run, are tested, and are
  wired end to end. The arbiter now enforces all three limit families the spec
  names — position size (gross cap, confidence scaling, quarter-Kelly), variance
  (vol targeting + an exactly-solved portfolio variance budget), and drawdown
  (hard halt + linear de-risking) — in closed-form NumPy with zero LLM logic,
  covered by 1,578 pytest cases in `tests/test_risk_arbiter.py`.
  The seam left open is `Generator.scorer`: a rank-IC heuristic stands where the
  learned model goes. The contract around it is what matters and it is fixed.
- **Phase 2 — built.** SCM with `do()` interventions and twin-network
  counterfactuals; free-energy belief updates driving action selection. Open
  seam: the causal graph is hand-specified, not fitted from data.
- **Phase 3 — partial.** Energy-based tensors exist (`F = U(1−TS)`, temperature,
  entropy, pressure). Quantum-inspired stochastic branching does **not** exist
  yet — `Multiverse` branches are classical and deterministic given a seed. Do
  not describe it as more than that.

## Standing instructions

- **Never report a confidence the data cannot justify.** Small samples get
  shrunk, not rounded up. Estimates say `est.` Simulated numbers say `SIMULATED`.
- **Determinism is not optional.** Every stochastic component takes an explicit
  seed. Two runs with the same seed produce identical output, or it is a bug.
- **Grand names are metaphors for ordinary math.** Keep the metaphor in the
  docstring and the honest mechanic in the code. A name must never oversell what
  a function does.
- **STARK holds no capital.** It emits rulings. It places no orders, holds no
  keys, and makes no network calls. Execution lives outside this tree.
- **Dependencies:** `numpy` for the engine, `pandas` + `streamlit` for the app.
  Adding one is an architecture decision, not a convenience.
- Every module runs standalone and prints a demo of what it does.
- Verify with `pytest stark-core/tests/ -q` before committing (the smoke file also
  runs standalone: `python3 stark-core/tests/test_smoke.py`).
- **A risk cap is approached from below, never from above.** Sizes are floored,
  not rounded — rounding 0.070080 up to 0.0701 breaches a budget, and "by a hair"
  is not a category the arbiter recognises.
