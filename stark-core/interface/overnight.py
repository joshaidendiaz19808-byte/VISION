#!/usr/bin/env python3
"""
Level 4 - overnight counterfactual runs.

JARVIS answers a macro question in milliseconds because the work was done at
03:00, not because anything here is fast. A scenario sweep across a causal model
takes minutes; a human asking "what happens if the Strait closes" will not wait
minutes, and a system that makes them wait gets used once.

So: batch the scenarios overnight, persist the results with their seeds and their
provenance, and let the interface stream from the store.

TWO PROPERTIES THIS FILE ENFORCES

  Nothing is fabricated on demand. If no overnight run covers the question, the
  store returns nothing and JARVIS says so. It never quietly interpolates
  between two other scenarios and presents the result as an answer.

  Results go stale. A counterfactual computed against Tuesday's book is not an
  answer about Thursday's. Every result records the book state it assumed, and
  `stale_against` compares. An old answer delivered confidently is worse than no
  answer, because it will be believed.

Every number produced here is SIMULATED, and every one of them is labelled.
Simulated evidence sizes nothing - the arbiter refuses it outright, by rule.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    import pathlib as _pl, sys as _sys
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

from simulation.causal_engine import CausalGraph, Multiverse


@dataclass(frozen=True)
class Scenario:
    """A question worth having an answer ready for."""

    scenario_id: str
    name: str
    description: str
    trigger_entity: str                      # which knowledge-graph node it starts at
    interventions: Mapping[str, float]       # do(...) on the causal model
    severity: str = "moderate"               # mild | moderate | severe

    def __str__(self) -> str:
        return f"{self.name} [{self.severity}]"


@dataclass(frozen=True)
class BranchOutcome:
    """One branch of one scenario. SIMULATED, always."""

    label: str
    outcome_mean: float
    outcome_p05: float
    outcome_p95: float
    regret_vs_baseline: float

    @property
    def spread(self) -> float:
        return self.outcome_p95 - self.outcome_p05


@dataclass(frozen=True)
class ScenarioResult:
    """What the overnight run established. Note what it is NOT: evidence."""

    scenario_id: str
    name: str
    trigger_entity: str
    baseline_mean: float
    branches: tuple[BranchOutcome, ...]
    outcome: str
    seed: int
    samples: int
    book_state_hash: str
    computed_at: float
    runtime_seconds: float
    simulated: bool = True                   # never anything else

    @property
    def worst(self) -> BranchOutcome:
        return min(self.branches, key=lambda b: b.outcome_mean)

    @property
    def best(self) -> BranchOutcome:
        return max(self.branches, key=lambda b: b.outcome_mean)

    @property
    def age_hours(self) -> float:
        return (time.time() - self.computed_at) / 3600.0

    def stale_against(self, book_state_hash: str, max_age_hours: float = 36.0) -> str | None:
        """Why this result should not be trusted right now, or None if it is fine."""
        if book_state_hash != self.book_state_hash:
            return ("computed against a different book state - the position has changed "
                    "since this ran, so the numbers describe a portfolio you no longer hold")
        if self.age_hours > max_age_hours:
            return f"computed {self.age_hours:.0f} hours ago, past the {max_age_hours:.0f}h limit"
        return None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["branches"] = [asdict(b) for b in self.branches]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping) -> "ScenarioResult":
        data = dict(payload)
        data["branches"] = tuple(BranchOutcome(**b) for b in data["branches"])
        return cls(**data)


def book_state_hash(positions: Mapping[str, float]) -> str:
    """A stable fingerprint of the book, so a stale answer can be spotted."""
    canonical = json.dumps({k: round(float(v), 6) for k, v in sorted(positions.items())})
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class OvernightStore:
    """
    Persisted scenario results. Written by the batch job, read by the interface.

    `find` returns nothing rather than the closest thing it has. "I have not run
    that" is a complete and useful answer; a neighbouring scenario presented as
    the requested one is not.
    """

    def __init__(self, results: Sequence[ScenarioResult] = ()) -> None:
        self.results: list[ScenarioResult] = list(results)

    def add(self, result: ScenarioResult) -> None:
        self.results = [r for r in self.results if r.scenario_id != result.scenario_id]
        self.results.append(result)

    def find(self, scenario_id: str) -> ScenarioResult | None:
        return next((r for r in self.results if r.scenario_id == scenario_id), None)

    def for_entity(self, entity_id: str) -> list[ScenarioResult]:
        return [r for r in self.results if r.trigger_entity == entity_id]

    def coverage(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for result in self.results:
            out[result.trigger_entity] = out.get(result.trigger_entity, 0) + 1
        return out

    def save(self, path: str | Path) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"results": [r.to_dict() for r in self.results]},
                                   indent=2, sort_keys=True))
        return len(self.results)

    @classmethod
    def load(cls, path: str | Path) -> "OvernightStore":
        payload = json.loads(Path(path).read_text())
        return cls([ScenarioResult.from_dict(r) for r in payload.get("results", [])])


class OvernightRunner:
    """
    The batch job. Deterministic given its seed - two nights with the same seed
    and the same model produce identical results, or something is wrong.
    """

    def __init__(self, graph: CausalGraph, outcome: str = "pnl_bps",
                 exposure_node: str = "exposure", samples: int = 4_000,
                 seed: int = 0, responses: Mapping[str, float] | None = None) -> None:
        self.graph = graph
        self.outcome = outcome
        self.exposure_node = exposure_node
        self.samples = int(samples)
        self.seed = int(seed)
        # The branches vary OUR RESPONSE under the shock, not the size of the
        # shock. "What if it is half as bad" is a weather forecast; "what should
        # we hold if it happens" is a decision, and only the second one is
        # actionable at 06:00 with a human waiting.
        self.responses = dict(responses or
                              {"flatten": 0.0, "halve": 0.5, "hold": 1.0, "add": 1.5})

    def run_one(self, scenario: Scenario, positions: Mapping[str, float]) -> ScenarioResult:
        started = time.perf_counter()
        multiverse = Multiverse(self.graph, outcome=self.outcome)
        shock = dict(scenario.interventions)
        has_exposure = self.exposure_node in self.graph.names
        scenarios = {"shock, no change": shock}
        if has_exposure:
            scenarios = {f"shock, {label}": {**shock, self.exposure_node: level}
                         for label, level in self.responses.items()}
        multiverse.explore(scenarios, n=self.samples, seed=self.seed)
        baseline = multiverse.baseline.mean(self.outcome)

        branches = []
        for branch in multiverse.branches:
            values = branch.world[self.outcome]
            branches.append(BranchOutcome(
                label=branch.label,
                outcome_mean=float(np.mean(values)),
                outcome_p05=float(np.percentile(values, 5)),
                outcome_p95=float(np.percentile(values, 95)),
                regret_vs_baseline=branch.regret_vs(multiverse.baseline),
            ))

        return ScenarioResult(
            scenario_id=scenario.scenario_id, name=scenario.name,
            trigger_entity=scenario.trigger_entity, baseline_mean=float(baseline),
            branches=tuple(branches), outcome=self.outcome, seed=self.seed,
            samples=self.samples, book_state_hash=book_state_hash(positions),
            computed_at=time.time(), runtime_seconds=time.perf_counter() - started,
        )

    def run_all(self, scenarios: Sequence[Scenario],
                positions: Mapping[str, float]) -> OvernightStore:
        store = OvernightStore()
        for scenario in scenarios:
            store.add(self.run_one(scenario, positions))
        return store


def demo_model() -> CausalGraph:
    """A toy energy book: transit capacity -> freight -> crude -> our P&L."""
    return (CausalGraph()
            .add("transit_capacity", (), lambda v, e: 1.0 + 0.15 * e)
            .add("freight_rate", ("transit_capacity",),
                 lambda v, e: 2.0 / np.maximum(v["transit_capacity"], 0.15) + 0.3 * e)
            .add("crude_bps", ("freight_rate",), lambda v, e: 8.0 * v["freight_rate"] + 12.0 * e)
            .add("refining_margin", ("transit_capacity",),
                 lambda v, e: 30.0 * v["transit_capacity"] - 10.0 + 6.0 * e)
            .add("exposure", (), lambda v, e: np.full_like(e, 1.0))
            .add("pnl_bps", ("crude_bps", "refining_margin", "exposure"),
                 lambda v, e: v["exposure"] * (0.6 * v["crude_bps"] + 0.4 * v["refining_margin"])
                 - 20.0 + 2.0 * e))


DEFAULT_SCENARIOS = (
    Scenario("malacca-closure-severe", "Strait of Malacca closure",
             "Transit capacity falls to 20% for an extended period",
             "malacca", {"transit_capacity": 0.2}, severity="severe"),
    Scenario("malacca-disruption-mild", "Strait of Malacca partial disruption",
             "Transit capacity falls to 70%", "malacca",
             {"transit_capacity": 0.7}, severity="mild"),
    Scenario("freight-spike", "VLCC freight rate spike",
             "Freight doubles on tonnage scarcity, transit unaffected",
             "crude_freight", {"freight_rate": 8.0}, severity="moderate"),
)


if __name__ == "__main__":
    print("STARK MK18 - overnight counterfactual runs\n")

    positions = {"crude": 0.06, "diesel_crack": 0.03}
    runner = OvernightRunner(demo_model(), samples=4_000, seed=11)

    print(f"running {len(DEFAULT_SCENARIOS)} scenarios against book "
          f"{book_state_hash(positions)}...\n")
    store = runner.run_all(DEFAULT_SCENARIOS, positions)

    for result in store.results:
        print(f"{result.name}  ({result.runtime_seconds * 1000:.0f} ms, "
              f"{result.samples:,} samples, seed {result.seed})")
        print(f"  baseline {result.baseline_mean:+.1f} bps   [ALL FIGURES SIMULATED]")
        for branch in sorted(result.branches, key=lambda b: b.outcome_mean, reverse=True):
            print(f"    {branch.label:<18} {branch.outcome_mean:+8.1f} bps  "
                  f"(p05 {branch.outcome_p05:+7.1f}, p95 {branch.outcome_p95:+7.1f}, "
                  f"regret {branch.regret_vs_baseline:+7.1f})")
        print()

    print("Every branch holds the SHOCK fixed and varies OUR RESPONSE. That is what")
    print("makes the run actionable: the question at 06:00 is not how bad it might be,")
    print("it is what to hold if it happens.\n")
    print(f"store coverage by entity: {store.coverage()}")

    stale = store.results[0].stale_against(book_state_hash({"crude": 0.20}))
    print(f"\nasked about a DIFFERENT book, the same result reports:\n  {stale}")
    print("\nThat check is the point. An overnight number delivered confidently against")
    print("a book that has since changed is worse than no number, because it will be")
    print("believed. And every figure above is SIMULATED: it sizes nothing.")
