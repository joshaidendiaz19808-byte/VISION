# STARK MK18 — Honest Assessment

Written after the full build. The through-line, stated once up front:

> **The engineering is sound. The validation is circular.**
> Every result in this repo was produced by feeding a model data generated from
> the same assumptions the model embodies. Nothing here has yet been wrong in
> the way real data makes you wrong.

**State:** 8,041 lines, 313 distinct test functions (1,863 parametrized cases),
15 runnable module demos, everything committed at `e7bda61`.

---

## What's genuinely good

### 1. The risk arbiter is the best thing in the repo
Dependency-free, closed-form NumPy, no learned thresholds. It imports nothing
from STARK, so no caller can hand it a clever object and walk past it. Nine
auditable predicates; permitted size is the minimum of every cap and the ruling
names which one bound.

It has now been tested against something that argues back. A deliberately
corrupted scorer returning 0.99 confidence on pure noise promoted six junk
features to the top of the queue; **zero of six** survived the red team, and the
arbiter vetoes them regardless of stated confidence.

### 2. The system refuses, and says why
Roughly fifteen distinct refusals across the codebase, each naming its reason
and usually the fix:

| Refuses | Because |
|---|---|
| A hypothesis under 100 observations | An anecdote is not a track record |
| Simulated evidence for sizing | Simulation sizes nothing |
| A macro value with no release date | That's look-ahead in a suit |
| A causal query down an unoriented edge | Correlation is not a mechanism |
| A question with no overnight run | "Not modelled" is a complete answer |
| A counterfactual against a changed book | A stale answer gets believed |
| A single-tank SAR reading at 10 m | The error bar exceeds the signal |
| An unapproved adjustment | A human signs, or nothing happens |

### 3. The tests check assumptions, not outputs
The integrator against analytic OU moments. The fitter by planting parameters
and recovering them inside their own error bars. Tensor contraction against
brute-force enumeration to 1e-12. This is why they caught **fifteen real bugs**
instead of just passing.

### 4. Four architecture calls that held up under pressure
- **One-way conviction valve** — conviction only ever falls, generator → red team → arbiter.
- **Search free, promote gated** — gating the ego loop's *search* by the arbiter made it unable to learn out of a bad regime (480/480 vetoed). The arbiter governs what gets *sized*, not what gets *thought*.
- **Whitelist composition, never `exec()`** — kernels change at runtime but every part was human-reviewed. Generated code can't be vetoed.
- **Projection, not penalty** — a penalty lets a constraint be bought off; projection makes the feasible set the only place the optimiser can stand.

---

## What's weak

Ranked by how much it matters.

### 1. Everything is synthetic — this is the whole problem
Sixteen modules run on generators I wrote. The SDE recovers parameters from data
produced by the same SDE. The red team kills noise features I generated as
noise. The scorer learned decay patterns from my own simulator. The dual loop
learned a regime I defined.

**The engine is verified. No edge is.** Those are different claims and the gap
between them is where trading systems die.

### 2. The test count flatters
1,863 sounds like ten times the coverage it is. It's **313 distinct test
functions**; 1,578 of the cases come from 30 parametrized functions, mostly one
384-proposal grid. The grid is genuinely useful. The headline number is not.

### 3. The learned scorer is weak in absolute terms
Rank correlation of **+0.25** against out-of-sample IC (heuristic: +0.19). Better,
and measured, but it's not a brain. I probed whether a bigger model or a ranking
objective helps — both are *worse*. Data is the constraint, not capacity.

### 4. Structure learning assumes no hidden confounders
The PC algorithm assumes **causal sufficiency** — that every common cause is
measured. In markets that assumption is nearly always false: an unmeasured
factor driving two variables produces a confident spurious edge that the
orientation rules will happily direct. The module reports Markov-equivalence
ambiguity honestly, but it cannot detect a confounder it was never shown. This
is the least-guarded assumption in the repo.

### 5. The knowledge graph is a seven-entity toy
Hand-built, with weights I chose. The *machinery* for refusing bad questions is
real and tested. The world model it refuses them against is a demo.

### 6. "24/7 ingestion" doesn't exist
`ingestion/` is a set of **libraries**, not running services. No scheduler, no
daemon, no retry queue, no monitoring, no alerting. Nothing runs unless someone
runs it. The arXiv crawler works and has fetched real papers; it has never run
twice on its own.

### 7. Python is not a reflex loop
The hot path measures ~10 µs p50. That's Python, not microseconds. What's
language-independent is the *contract* — fixed work per tick, no allocation in
the decision, atomic swap. The speed claim would need C or Rust.

### 8. Smaller things
- **No CI.** Tests run when someone remembers to run them.
- **Single-process assumptions.** Session state and the overnight store have no locking.
- **The ego loop optimizes a proxy that rewards size** — it inflated 0.02 → 0.15 until the arbiter stopped it. Goodhart, working as advertised.
- **No cross-sectional generator.** One feature against one forward return; real books are multi-asset.
- **No venue-calibrated cost model.** The square-root law is right in shape, uncalibrated in level.

---

## What's missing entirely

| Missing | Why it's missing |
|---|---|
| Real training corpus | Needs your data |
| Live feeds (imagery, AIS, news) | Needs credentials and contracts |
| Broker adapter | Deliberate — that's a decision with your signature |
| Scheduler / orchestration | Not built |
| CI pipeline | Not built |
| Audio in the "voice" layer | Not built; it's a text channel and says so |
| Any neural network | Rule 1 says "neural networks"; what exists is boosted trees, named accurately |

---

## What I'd work on, in order

**1. One real dataset, end to end.** Not a phase — an afternoon. Run the
generator on real features, let the red team attack them, see what survives. My
expectation is *almost nothing survives*, and that would be the first result in
this project not guaranteed by construction.

**2. Real corpus intake for the scorer.** A `corpus_from_history()` so the day
you point it at real feature history, it refits and the 29% number stops being
me grading my own homework.

**3. CI.** 313 tests that nobody runs automatically will rot. This is an hour.

**4. Confounder sensitivity in structure learning.** At minimum, report how
strong an unmeasured common cause would have to be to overturn each discovered
edge. Standard sensitivity analysis, and it directly guards the weakest
assumption in the repo.

**5. Then** — and only with numbers in hand — decide about live feeds, a
faster reflex loop, or a bigger model. All three are currently answers to
questions nobody has measured.

---

## The one-paragraph version

What exists is a careful, well-tested engine that reasons correctly and refuses
honestly, with a risk layer that has held against everything thrown at it
including a model built to lie to it. What does not exist is any evidence that
its conclusions mean anything about markets, because it has never seen one. The
next step is not another phase. It is one real dataset.
