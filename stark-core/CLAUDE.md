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
Level 2  simulation/    causal multiverse + active inference (what WOULD happen)
Level 1  core/          generate → attack → arbitrate       (what we DO)
```

**Dependency rule (enforced, not aspirational):** imports flow *downward only*.

| Package       | May import from                       |
|---------------|---------------------------------------|
| `core/`       | stdlib, numpy, other `core/` modules  |
| `simulation/` | the above + `core/`                   |
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
- Dependencies: `numpy` only. Adding a dependency is an architecture decision,
  not a convenience.

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

## 7. The MK18 loop

```
tensors      → field state (energy, entropy, temperature, disequilibrium)
generator    → hypotheses, ranked by shrunk intuition
red_team     → adversarial attacks; survivors only
risk_arbiter → ALLOW / CLAMP / VETO + a hard size cap
causal_engine→ branch the survivors under do(), measure regret
active_inf.  → surprise → belief update → what to test next
jarvis_node  → narrates the whole thing in plain English
```

Each stage may only ever *reduce* conviction. Nothing downstream of the red team
is allowed to talk conviction back up.

## 8. Verification

`python3 stark-core/tests/test_smoke.py` (or `pytest stark-core/tests`).

The suite is not only a smoke test - it enforces the rules above in code:
the veto is unappealable, confidence only ever falls, the arbiter imports
nothing from STARK, nobody touches the global RNG, simulated numbers are
labelled, and `F` never flips the sign of `U`. If you relax a rule here, a test
turns red. That is the intended failure mode.

## 9. Status: MK18 scaffold

What exists is a complete, runnable skeleton with honest maths in every joint -
each module runs, is tested, and does what its docstring says. What does NOT
exist yet: a real learned scorer behind `Generator.scorer` (currently a rank-IC
heuristic), a causal graph fitted to real data (currently hand-specified), a
live data feed, and any audio in the "voice" layer (currently a text channel).
Those are the four seams to build into. None of them change the architecture.
