# STARK MK18 — standing instructions & architecture rules

STARK is VISION's engine room. VISION answers *"is this strategy real or fake?"*
STARK answers the harder one: **"what should we believe next, and what are we
allowed to do about it?"**

Read this file before touching anything in `stark-core/`.

---

## 1. The four levels

```
Level 4  interface/     voice + spatial causal streaming   (how a human hears it)
Level 3  tensors/       thermodynamic equilibrium fields   (what the market IS)
Level 2  simulation/    causal engine + structure learning + active inference
Level 1  core/          generate → attack → arbitrate       (what we DO)
```

**Dependency rule (enforced, not aspirational):** imports flow *downward only*.

| Package       | May import from                       |
|---------------|---------------------------------------|
| `core/`       | stdlib, numpy, other `core/` modules  |
| `simulation/` | the above + `core/` + scipy           |
| `tensors/`    | stdlib, numpy **only** (it is substrate — it depends on nothing) |
| `interface/`  | anything below it                     |

If you ever need an upward import, the abstraction is wrong. Push the shared
type *down*, don't reach *up*.

`core/risk_arbiter.py` is deliberately dependency-free — it imports nothing from
STARK at all. It takes plain primitives so no other module can smuggle a
custom object past it.

## 2. The veto is final

`risk_arbiter` is the only place hard limits live. Nothing else in the codebase
may define a position cap, a leverage limit, or a "well, just this once".

- A `VETO` is absolute. Neural confidence, simulated upside, and a good story
  are all irrelevant to it.
- `RiskVeto` must never be caught and downgraded. Catch it only to *report* it,
  then re-raise or stop.
- The arbiter is symbolic on purpose: every rule is a readable predicate a human
  can audit in one line. No learned thresholds, ever.

## 3. Honesty rules (inherited from VISION)

- **Never report a confidence the data cannot justify.** Small samples get
  shrunk, not rounded up. `n=12` is not evidence.
- **Multiple testing is not optional.** If the generator looked at 40 features,
  the red team corrects for 40. Searching hard is *not* the same as being right.
- Simulated numbers are labelled simulated. Estimated numbers say `est.`
- If a module cannot answer, it says so. It does not interpolate a vibe.
- Grand names (`holodeck`, `multiverse`, `thermodynamic`) are **metaphors for
  ordinary math**. Keep the metaphor in the docstring and the honest mechanic in
  the code. Never let the name oversell the function.

## 4. Determinism

Every stochastic component takes an explicit `seed`. No bare `np.random.*`
anywhere. Two runs with the same seed produce byte-identical output, or it is a
bug. Reproducibility is what makes a red-team result an argument instead of an
anecdote.

## 5. Hard boundaries

- STARK **never** places an order, holds a key, or talks to a broker. It emits
  *rulings*; execution lives outside this tree and is somebody else's blast radius.
- No network calls inside `stark-core/`. Data arrives as arrays passed in.
- Dependencies: `numpy` everywhere; `scipy` in `simulation/` only. `tensors/`
  and `core/risk_arbiter.py` are numpy-only and stay that way. Adding a
  dependency is an architecture decision, not a convenience.
- Euler-Maruyama is weak order 1: keep `theta * dt <= 0.02` for a mean-reverting
  process, or the integrator biases the very thing you are trying to measure.

## 6. Conventions

- Every module runs standalone: `python stark-core/<pkg>/<module>.py` prints a
  self-contained demo of what it does.
- `stark-core/` itself is the import root (the folder name is hyphenated, so it
  is a path root, not an importable package). Put it on `sys.path`, then import
  `core.generator`, `tensors.market_tensors`, etc. A module that imports a
  sibling package carries this three-line guard so it still runs standalone:

  ```python
  if __package__ in (None, ""):
      import pathlib as _pl, sys as _sys
      _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
  ```
- Dataclasses for anything crossing a module boundary. Dicts are for internals.
- Type hints on every public function. Docstrings say *what it honestly does*,
  not what it evokes.

## 7. Two rules Phase 3 added

- **Hard limits are enforced by PROJECTION, never by a penalty term.** A penalty
  lets an optimiser buy its way past a constraint if the objective wants it
  badly enough. `project_to_capped_simplex` makes the feasible set the only
  place the solver may stand.
- **A grand name buys nothing.** "Quantum-inspired" here means the math was
  borrowed from quantum many-body physics — no superposition, no quantum
  hardware, no quantum speedup. Say so in the docstring, every time.

## 8. The MK18 loop

```
tensors      → field state (energy, entropy, temperature, disequilibrium)
             → capital equilibrium under hard caps; probability cloud of futures
generator    → hypotheses, ranked by shrunk intuition
red_team     → adversarial attacks; survivors only
risk_arbiter → ALLOW / CLAMP / VETO + a hard size cap
causal_engine→ branch the survivors under do(), measure regret
structure    → discover the graph; name the experiment where data cannot decide
active_inf.  → surprise → belief update → what to test next (offline and live)
jarvis_node  → narrates the whole thing in plain English
```

Each stage may only ever *reduce* conviction. Nothing downstream of the red team
is allowed to talk conviction back up.

## 9. Verification

`pytest stark-core/tests/ -q` — 1,706 cases. The smoke file also runs standalone:
`python3 stark-core/tests/test_smoke.py`.

`tests/test_risk_arbiter.py` is the Phase 1 guardrail suite: closed-form checks
on the math, one case per way a proposal can be invalid, and a 384-proposal grid
asserting that no input — including nonsense — can produce a size that is
negative, NaN, above the gross cap, larger than requested, or in breach of the
variance budget.

`tests/test_simulation.py` is the Phase 2 suite. It tests assumptions rather
than outputs: the integrator against analytic OU moments, the fitter by planting
parameters and recovering them within their own error bars, each model against
the physical claim it makes (delay delays, spreads widen with flow, cost is
concave in size, a longer lead time hurts more than a demand shock), branches
against the requirement that they differ by the action and not by luck, and the
filter against the frozen model it claims to beat.

`tests/test_structure_learning.py` checks that discovery finds the true graph
where it can AND leaves a chain undirected where it must — an algorithm that
always returns a fully oriented graph is broken, not confident.

`tests/test_tensors.py` checks the equilibrium solver against the analytic
Boltzmann solution, checks that no temperature crosses a hard cap, and checks
every contraction result against brute-force enumeration to 1e-12.

The suite is not only a smoke test - it enforces the rules above in code:
the veto is unappealable, confidence only ever falls, the arbiter imports
nothing from STARK, nobody touches the global RNG, simulated numbers are
labelled, and `F` never flips the sign of `U`. If you relax a rule here, a test
turns red. That is the intended failure mode.

## 10. Status: MK18

All three phases are built, tested, and wired end to end. What does NOT exist:
a real learned scorer behind `Generator.scorer` (still a rank-IC heuristic), a
live data feed, and any audio in the "voice" layer (still a text channel).
Those are the seams that remain. None of them change the architecture.
