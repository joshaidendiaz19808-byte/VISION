# Project STARK: Autonomous AGI Capital & Simulation Organism

## Core Architecture Rules
1. **Separation of Concerns:** Neural networks (LLMs) handle intuition, pattern matching, and hypothesis generation. They are *never* trusted with absolute execution logic.
2. **Deterministic Safety:** All risk boundaries, invariant checks, and vetoes must live in pure, deterministic Python/NumPy code (`risk_arbiter.py`). No LLM can override a hard mathematical constraint.
3. **Active Inference over Rewards:** Optimization loops must minimize prediction error and free energy rather than relying on standard reinforcement learning reward functions.

## Build Phases
- **Phase 1:** Build the 3-node adversarial core (Generator -> Red Team -> Risk Arbiter).
- **Phase 2:** Implement the Generative Causal Simulator and Active Inference loop.
- **Phase 3:** Scale up to Energy-Based Tensors and Quantum-Inspired Stochastic Branching.
- **Phase 4:** Omnipresent sensory ingestion and dual-loop execution.
- **Phase 5:** The J.A.R.V.I.S. interface — persistent, context-aware, and
  structurally unable to act without a named human's approval.

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
- **Phase 2 — built.** Two engines. The discrete SCM (`do()` interventions,
  twin-network counterfactuals) plus a continuous-time SDE layer: order book
  microstructure (flow-driven permanent impact, Roll noise, flow-widened
  spreads, square-root execution cost) and a stochastic delay differential
  equation for supply chains. Parameters are fitted by maximum likelihood in
  SciPy with real error bars, not hand-specified. `PredictionErrorLoop` closes
  the real-time loop: predict one step ahead, observe, correct — on a regime
  shift it cuts RMSE 21% and cumulative free energy 10x against the same model
  frozen. `structure_learning.py` closes the last seam: the graph itself is
  discovered (PC skeleton → collider orientation → Meek propagation), and where
  observation cannot orient an edge it says so and names the experiment —
  interventions through `do()` finish the job.
- **Phase 3 — built.** Three layers in `tensors/`. Energy-based fields
  (`F = U(1−TS)`, temperature, entropy, pressure). `CapitalField` relaxes
  capital, hypotheses and risk limits to thermodynamic equilibrium by minimizing
  `F = U − TS` under hard caps enforced by projection — it reproduces the
  analytic Boltzmann solution where one exists, and no temperature can buy a
  weight past its cap. `MarketStateNetwork` contracts a tensor network over
  future market states: 2^40 joint configurations queried in milliseconds
  without enumerating one, verified exact against brute force.
  **On "quantum-inspired":** the math (matrix product states, bond dimension,
  transfer matrices, correlation length) is borrowed from quantum many-body
  physics and that is the entire borrowing. No superposition, no entanglement,
  no quantum hardware, no quantum speedup — the win is structural, from chain
  contraction being O(N·d²) instead of d^N. Exact for chains and trees only;
  loopy graphs would need an approximation this file does not implement. Do not
  describe it as more than that.
- **Phase 4 — built.** `ingestion/` carries the senses: a polite cached HTTP
  layer with mandatory provenance, a live arXiv crawler, point-in-time macro
  series that refuse to hand out numbers before their release date, SAR
  floating-roof tank geometry with honest error propagation, and AIS decoding
  with gap screening. `execution/dual_loop.py` carries the hands: a deterministic
  reflex loop on the hot path and an ego loop that genuinely restructures
  kernels at runtime — composing whitelisted operators, never generating code —
  behind four gates, one of which is the risk arbiter itself. **No imagery
  pipeline, no AIS feed, no news scraper, and no live broker adapter.** Those
  need credentials, contracts and a human signature, and none of them is
  written here.
- **Phase 5 — built.** `knowledge_graph.py` is JARVIS's memory: a typed graph
  where every edge records how it came to be believed, and causal queries may
  only walk edges whose *direction* was established. A measured correlation is
  stored, visible, and unwalkable. `overnight.py` precomputes counterfactuals so
  a macro question is answered in milliseconds, and every result records the
  book it assumed so a stale answer is refused rather than delivered.
  `jarvis_node.py` gains the console: a spatial dashboard, keyword entity
  resolution (called that, because it is not language understanding), a streamed
  causal breakdown, and exactly one recommendation behind an approval gate that
  only a named human opens. It refuses four ways — unknown entity, unoriented
  edge, no overnight run, stale book — and a veto produces no recommendation
  rather than a softened one.

## Standing instructions

- **Never report a confidence the data cannot justify.** Small samples get
  shrunk, not rounded up. Estimates say `est.` Simulated numbers say `SIMULATED`.
- **Determinism is not optional.** Every stochastic component takes an explicit
  seed. Two runs with the same seed produce identical output, or it is a bug.
- **Grand names are metaphors for ordinary math.** Keep the metaphor in the
  docstring and the honest mechanic in the code. A name must never oversell what
  a function does.
- **STARK holds no capital.** It emits rulings. It places no orders, holds no
  keys, and makes no network calls. Execution lives outside this tree — in
  `execution/`, which emits *intents*; the only adapter records them and sends
  nothing. Ingestion, which does touch the network, lives in `ingestion/`.
- **The arbiter governs what gets SIZED, not what gets THOUGHT.** Gating a
  search by a risk limit is the same category error as letting a hypothesis
  override one, and it fails just as badly — quietly, by never improving.
- **No generated code is ever executed.** The ego loop rewrites kernels by
  composing whitelisted operators, never by `exec()`-ing source. Code nobody has
  read cannot be reviewed before it trades or reproduced after it does.
- **Dependencies:** `numpy` throughout, `scipy` for the simulation layer only
  (integration, MLE, linear algebra), `pandas` + `streamlit` for the app.
  `tensors/` and `core/risk_arbiter.py` stay numpy-only — the substrate and the
  veto carry no optional weight. Adding a dependency is an architecture
  decision, not a convenience.
- Every module runs standalone and prints a demo of what it does.
- Verify with `pytest stark-core/tests/ -q` before committing (the smoke file also
  runs standalone: `python3 stark-core/tests/test_smoke.py`).
- **A risk cap is approached from below, never from above.** Sizes are floored,
  not rounded — rounding 0.070080 up to 0.0701 breaches a budget, and "by a hair"
  is not a category the arbiter recognises.
