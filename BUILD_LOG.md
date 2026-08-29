# STARK MK18 — Build Log

Everything built in this session, what it does, what it does *not* do, and every
bug the work surfaced. Written to be read cold, months from now, by someone who
has forgotten the details — including you.

**Status:** all three phases built, tested, wired end to end.
**Tests:** 1,706 passing (`pytest stark-core/tests/ -q`).
**Branch:** `claude/stark-mk18-scaffold-xnc72k`
**Engine:** ~3,800 lines across 8 modules, each runnable standalone.

---

## Contents

1. [What got built](#1-what-got-built)
2. [Phase 1 — the Cognitive Nucleus](#2-phase-1--the-cognitive-nucleus)
3. [Phase 2 — continuous time and real-time inference](#3-phase-2--continuous-time-and-real-time-inference)
4. [Structure discovery](#4-structure-discovery)
5. [Phase 3 — tensor fields and networks](#5-phase-3--tensor-fields-and-networks)
6. [Every bug found and fixed](#6-every-bug-found-and-fixed)
7. [Honesty decisions](#7-honesty-decisions-and-why-each-one-matters)
8. [Test suite](#8-the-test-suite)
9. [What does not exist](#9-what-does-not-exist)
10. [How to run it](#10-how-to-run-it)

---

## 1. What got built

```
VISION/
├── CLAUDE.md              project-wide law: 3 architecture rules, 3 build phases
├── MASTER_SPEC.md         the full target-state specification
├── BUILD_LOG.md           this file
├── app.py, src/vision.py  the original strategy validator (4 honesty tests)
└── stark-core/            the engine
    ├── CLAUDE.md          STARK's own rules: layers, imports, conventions
    ├── core/              Phase 1  generator → red_team → risk_arbiter
    ├── simulation/        Phase 2  causal_engine, structure_learning, active_inference
    ├── tensors/           Phase 3  market_tensors (fields + networks)
    ├── interface/         Level 4  jarvis_node
    └── tests/             1,706 cases
```

| Module | Lines | Job |
|---|---:|---|
| `core/generator.py` | 236 | Proposes falsifiable claims, confidence shrunk by sample size |
| `core/red_team.py` | 264 | Six adversarial attacks; survivors only |
| `core/risk_arbiter.py` | 429 | Hard symbolic guardrails — the veto is final |
| `simulation/causal_engine.py` | 839 | Discrete SCM + continuous-time SDEs |
| `simulation/structure_learning.py` | 422 | Discovers the causal graph from data |
| `simulation/active_inference.py` | 461 | Free energy, belief updates, real-time filtering |
| `tensors/market_tensors.py` | 881 | Thermodynamic fields, capital equilibrium, tensor networks |
| `interface/jarvis_node.py` | 289 | Narrates the whole chain in plain English |

**The dependency rule, enforced by tests:** imports flow downward only.
`core/` takes stdlib + numpy. `simulation/` adds scipy and may import `core/`.
`tensors/` is numpy-only — it is the substrate and depends on nothing.
`core/risk_arbiter.py` imports nothing from STARK at all.

---

## 2. Phase 1 — the Cognitive Nucleus

**Generator → Red Team → Risk Arbiter.** Each stage may only ever *reduce*
conviction. Nothing downstream of the red team can talk it back up.

### The generator
Ranks features against forward returns by rank IC, then shrinks the result by
sample size — `n=10` keeps ~17% of the signal, `n=250` keeps ~83%. Everything it
emits is a **prior**, never evidence. It records how many features it searched
so the red team can charge for the search.

`Generator.scorer` is a pluggable hook: a rank-IC heuristic stands where a
learned model goes. The contract around it is fixed.

### The red team — six attacks
| Attack | What it catches |
|---|---|
| out-of-sample | Edge that inverts or decays on unseen data |
| stability | Sign flipping across folds — one lucky window |
| multiple testing | Šidák correction: searching hard is not being right |
| permutation | Shuffled data reproducing the result |
| cost | Edge smaller than the toll to harvest it |
| regime | Edge that exists in only one volatility state |

Live demo result — four features, one real signal plus RSI, Twitter sentiment
and moon phase. The three fakes die (moon phase killed by out-of-sample
inversion, the search penalty, permutation, and cost); the real signal survives
at 0.11 confidence.

### The risk arbiter
**Zero LLM logic.** No model, no prompt, no learned threshold, no sampling.
Every limit is a hard-coded constant and every decision is closed-form NumPy.
An `ast`-based test allowlists its imports to numpy and stdlib, so the module's
own prose about having no LLM cannot fool the check.

Three limit families:

- **Position size** — gross cap (25%), confidence scaling, quarter-Kelly
- **Variance** — vol targeting, plus a portfolio variance budget **solved, not
  searched**: portfolio variance is a quadratic in the new weight, so the largest
  admissible size is closed-form and lands exactly on the 150 bps budget
- **Drawdown** — hard halt at 20%, linear de-risking on the way down

Nine auditable predicates. Hard hits → `RiskVeto`. Soft hits → clamp. Permitted
size is the **minimum** of every cap, and the ruling names which one bound.

```
book already carrying most of its variance budget:
  CLAMP - max size 0.070 of book (bound by variance-budget, book vol 150 bps)
```

The two tests that matter most:
- `test_maximum_confidence_cannot_buy_a_veto_off` — confidence 0.9 through 5.0
  against a 3-observation sample, all vetoed. **The neural half gets no vote.**
- A 384-proposal grid proving no input — including nonsense — produces a size
  that is negative, NaN, above the gross cap, larger than requested, or in
  breach of the variance budget.

---

## 3. Phase 2 — continuous time and real-time inference

### The SDE layer
Itô processes integrated by Euler–Maruyama, sitting beside the discrete SCM
(kept and still tested). Brownian increments are **retained**, so a
continuous-time counterfactual works: replay identical noise under a different
action and the branches differ by the action, not by luck.

**Order book microstructure** — four coupled SDEs:

```
Q  order flow imbalance   dQ = -κ_q Q dt + σ_q dW
E  efficient price        dE = (λQ + λ_own·u) dt + σ_e dW
M  observed mid           dM = κ_m (E - M) dt + η dW        ← Roll noise
S  half-spread            dS = κ_s(S̄ + γ|Q| - S) dt + σ_s dW
```

Imbalanced flow moves the efficient price permanently (Almgren–Chriss). The
observed mid *chases* the efficient price rather than being it — which is why
naive mid-price returns overstate what you can capture. Spreads widen exactly
when flow is imbalanced, so cost arrives when it hurts most. Our participation
is an intervention channel:

```
passive 0%        mid  -5.3 bps | cost  0.00 bps
gentle 5%         mid  -2.8 bps | cost 11.31 bps
aggressive 20%    mid  +4.6 bps | cost 24.00 bps
reckless 50%      mid +19.4 bps | cost 41.90 bps
```

Execution cost follows the **square-root law** — total cost concave in size,
cost per unit falling. This is one of the most robustly replicated facts in
market microstructure, and it is not a detail: linear impact says doubling your
size doubles your cost, square-root says it costs 1.41x. Get it wrong and every
optimal-size calculation downstream is wrong with it.

**Supply chain delay** — a stochastic *delay* differential equation. Goods
ordered at `t` arrive at `t+τ`, under a base-stock policy. The result worth
having: lengthening lead time 45d→75d hurts worst-case inventory **8x more**
than a +30% demand shock. That fell out of the mechanism; it was not put there.

**Calibration** — maximum likelihood in SciPy, not hand-specified parameters.
`fit_ornstein_uhlenbeck` uses the exact transition density with a closed-form
AR(1) starting point; standard errors come from a finite-difference
observed-information matrix.

```
generated with: theta=3.0000, mu=1.5000, sigma=0.8000
recovered:      theta=3.3087±0.3961, mu=1.5306±0.0389, sigma=0.8131±0.0092
```

Every true value inside its error bar. That is the test a fitter has to pass
before it goes near data whose answer you do not already know.

### Real-time prediction error
`PredictionErrorLoop` predicts one step ahead, observes, measures the error,
corrects. It adapts an explicit Gaussian belief over model bias rather than
refitting, so the adaptation stays **one auditable number**. Coupling is by
`Protocol`, not import, so the two Level 2 modules stay independent.

A backtest fits parameters on history and freezes them. It cannot notice the
world moved, because nothing in it is looking. On a regime shift, same model,
same data, run two ways:

```
static (backtest)  1200 steps | RMSE 0.2283 | mean surprise 0.978  ← DRIFTING
online (filter)    1200 steps | RMSE 0.1792 | mean surprise 0.097

RMSE 21% lower | cumulative free energy +1174 → +116
```

---

## 4. Structure discovery

The Phase 2 seam: the causal graph is now **learned, not drawn**. PC skeleton
via partial-correlation independence tests → collider orientation from
separating sets → Meek rules R1–R3 for propagation.

**The honest half is the point.** Observational data identifies a graph only up
to its Markov equivalence class. `A→B→C` and `C→B→A` imply *identical* joint
distributions — no amount of watching separates them.

Case 1, a collider (`liquidity → flow ← news`, `flow → price`): solved by
observation alone, because a collider leaves a fingerprint nothing else does —
two independent causes become *dependent* once you condition on their shared
effect.

Case 2, a chain (`supply → inventory → spot`): not solvable by watching, and it
says so.

```
3 variables, 2 edges (0 oriented, 2 undecided)
     inventory -- spot      direction not identified by this data
     inventory -- supply    direction not identified by this data

what the data cannot answer:
  Is it inventory → spot or spot → inventory?
  Intervene on inventory: if spot moves, the arrow points at spot.

and it refuses to pretend otherwise:
  2 edge(s) still undirected. Orient them with interventions before building an SCM.
```

`to_causal_graph()` **raises** rather than picking a direction to make the code
run. Then interventions finish it: forcing supply moves inventory; forcing
inventory leaves supply alone. That asymmetry is the arrow. If forcing *either*
moves the other — hidden common cause or feedback — the edge stays undirected,
which is the correct answer, not a shortfall.

> This is why STARK simulates. An intervention in the world costs money and may
> be impossible; an intervention in the model costs a random seed.

---

## 5. Phase 3 — tensor fields and networks

### CapitalField — thermodynamic equilibrium
Three fields over one index set: **capital flows** (weights), **agent
hypotheses** (claimed edge), **risk limits** (hard caps). Relaxed to equilibrium
by minimizing

```
F = U - T·S     U(w) = -edge'w + (γ/2) w'Σw      S(w) = -Σ w log w
```

It reproduces the analytic **Boltzmann** solution `w ∝ exp(edge/T)` where one
exists — which is how we know the solver minimizes what it claims to minimize.

Temperature is the risk-appetite dial: heat diversifies, cold concentrates.
Neither is right — the dial is risk appetite stated as a number instead of a mood.

**Hard caps are enforced by projection onto the capped simplex, never a penalty.**
A penalty lets a constraint be bought off if the objective wants it badly
enough, which is precisely what the arbiter exists to forbid. In the demo, the
strongest hypothesis in the book (22 bps edge) sits pinned at its 10% cap at
*every* temperature.

### MarketStateNetwork — the probability cloud
A chain tensor network over discrete future market states. One contraction
gives marginals, conditionals, exact top-k joint futures, exact ancestral
samples, and transfer-matrix correlation length.

```
News arrives: liquidity observed THIN. The whole cloud updates at once.
  regime      calm 57.3%→27.8%   stressed 42.7%→72.2%
  direction   down 45.0%→51.7%   flat 29.0%→25.8%   up 26.0%→22.5%

40 variables: 1,099,511,627,776 joint configurations
  time taken: 3.9 ms, without enumerating a single configuration
```

Verified **exact** against brute-force enumeration to 1e-12 — not a sampling
approximation, not mean-field.

### On "quantum-inspired" — stated plainly in the code
The math (matrix product states, bond dimension, transfer matrices, correlation
length) is borrowed from quantum many-body physics, and **that is the entire
borrowing.**

- No superposition, no entanglement, no quantum hardware, **no quantum speedup**
- The numbers are ordinary non-negative probabilities; every operation is
  classical linear algebra
- The win is **structural**: chain contraction is O(N·d²) instead of d^N
- Exact for chains and trees only. Loopy graphs need an approximation this file
  **does not implement**

This is written into both `CLAUDE.md` files so no future session can inflate it.

---

## 6. Every bug found and fixed

Each was found by a test or a demo, and fixed in the **code**, not worked around.

| # | Where | The bug | The fix |
|---|---|---|---|
| 1 | `market_tensors` | `F` inverted the sign of `U` in hot tapes — "stretched high" flipping to "stretched low" | Clamp the discount to [0,1]: F can shrink a stretch to nothing, never invert it |
| 2 | `risk_arbiter` | Size **rounded** to 4dp, rounding a 0.070080 variance solution *up* to 0.0701 — breaching the budget by a hair | Floor, never round. A risk cap is approached from below |
| 3 | `causal_engine` | Base-stock policy never replaced demand — ordering fell to zero exactly at target and inventory drained to −157 while the position looked healthy | Order `demand + gap`, not just `gap` |
| 4 | `causal_engine` | `L-BFGS-B`'s `hess_inv` came back near-identity, reporting `stderr == parameter` for every fit — an error bar that looked real and was not | Standard errors from a finite-difference Hessian; now match asymptotic theory to two digits |
| 5 | `causal_engine` | `SDEMultiverse.ranked()` passed a branch where its key expected a path | Key on `b.path` |
| 6 | `causal_engine` | Execution cost linear in size, and the demo called it "superlinear" — both wrong | Square-root law, with the claim corrected |
| 7 | `market_tensors` | `_logsumexp` broke on scalar reduction under NumPy 2 | Separate scalar path |
| 8 | `market_tensors` | Backtracking that only ever shrank the step turned the solver into a crawl — one early overshoot and every later step paid for it | Grow the step after a success |
| 9 | `market_tensors` | Convergence judged on weight movement reported *failure* at low temperature on a perfectly good answer — the entropy gradient's log term diverges as a weight → 0 | Converge on ΔF, the quantity actually being minimized. Runtime 31s → 10s |
| 10 | `active_inference` | Demo narrated a conclusion its own numbers contradicted | Generate the explanation from the actual choice |

**One test was wrong, not the code.** I asserted the filter should learn a bias
of `>1.0` after a +5 offset. It learned 0.244 — and 0.244 is *correct*:
the prediction starts from the observed state, which already carries the
offset, so what survives one step is `5·(1−e^{−θdt}) = 0.2439`. A filter that
learned +5 would be double-counting. The test now asserts the exact value.

Similarly, when a parameter-recovery test failed at θ=0.5, I checked the
estimator across 12 seeds before concluding anything: mean 0.510 against a truth
of 0.5, sd 0.070. The estimator was unbiased; seed 11 was a 1.8σ draw and my
tolerance was too tight. The test now judges recovery against the fit's own
error bars — the statistically correct criterion.

---

## 7. Honesty decisions, and why each one matters

- **`MASTER_SPEC.md` carries a target-state banner.** The spec reads as
  present-tense description ("bounces radar pulses", "counts container ships").
  What exists is the engine below. The banner points at `CLAUDE.md` for what is
  real, and the two files can no longer contradict each other.
- **The "63 passing invariant tests verified" claim is footnoted.** There are no
  spacetime-metric tests in this repo. Left in your wording, annotated as intent
  rather than fact.
- **A documented limitation instead of a hidden one.** Euler–Maruyama at
  θ·dt = 0.08 biases σ ~4% — enough to miss a planted parameter by 7σ. That is
  the *integrator*, not the fitter. There is a test asserting the bias exists at
  coarse dt and shrinks at fine dt, and the constraint (θ·dt ≤ 0.02) is in the
  engine docstring.
- **Simulated numbers say `SIMULATED`. Estimates say `est.`** Enforced by test.
- **No bare global RNG anywhere.** Every stochastic component takes an explicit
  seed; same seed, identical output. Enforced by test.
- **Grand names are metaphors for ordinary math.** Kept in the docstring, honest
  mechanic in the code. A name must never oversell what a function does.

---

## 8. The test suite

```
pytest stark-core/tests/ -q          →  1,706 passed
```

| File | Cases | What it pins down |
|---|---:|---|
| `test_risk_arbiter.py` | 1,578 | Math, 12 kinds of invalid proposal, clamps, a 384-proposal invariant grid, purity |
| `test_simulation.py` | 43 | Integrator vs analytic moments, parameter recovery, each model's physical claim, branching, the filter |
| `test_tensors.py` | 36 | Boltzmann agreement, caps unbreakable at any temperature, contraction vs enumeration |
| `test_smoke.py` | 28 | Every module imports, runs, is deterministic; architecture rules hold |
| `test_structure_learning.py` | 21 | Finds the true graph where it can, admits ambiguity where it must |

The suites test **assumptions, not outputs**: the integrator against known OU
moments, the fitter by planting parameters and recovering them within their own
error bars, each model against the physical claim it makes, contraction against
brute force, and the filter against the frozen model it claims to beat.

---

## 9. What does not exist

Stated plainly so nothing here gets oversold:

- **No learned model.** `Generator.scorer` is a rank-IC heuristic behind a
  pluggable hook. The contract is fixed; the model is not written.
- **No live data.** Everything arrives as arrays passed in. No network calls
  anywhere in `stark-core/`.
- **No audio.** The "voice" layer is a tagged text channel.
- **No execution.** STARK emits rulings. It places no orders, holds no keys.
- **No loopy tensor networks.** Chains and trees are exact; loops would need an
  approximation that is not implemented.
- **No satellites, no ingestion engine, no autonomous R&D.** Those are
  `MASTER_SPEC.md` target state, not built.

---

## 10. How to run it

```bash
pip install -r requirements.txt          # numpy, scipy, pandas, streamlit

pytest stark-core/tests/ -q              # 1,706 tests
python3 stark-core/tests/test_smoke.py   # smoke file also runs standalone

# every module prints a demo of what it does
python3 stark-core/core/generator.py
python3 stark-core/core/red_team.py
python3 stark-core/core/risk_arbiter.py
python3 stark-core/simulation/causal_engine.py
python3 stark-core/simulation/structure_learning.py
python3 stark-core/simulation/active_inference.py
python3 stark-core/tensors/market_tensors.py
python3 stark-core/interface/jarvis_node.py      # the whole chain, narrated
```

---

## The rule that holds the whole thing together

> Each stage may only ever **reduce** conviction.
> Nothing downstream of the red team is allowed to talk conviction back up.
> The veto is unappealable. Simulated evidence sizes nothing.
> A sub-100-observation track record is an anecdote.

These limits live in exactly one file — `core/risk_arbiter.py` — and nowhere else.
