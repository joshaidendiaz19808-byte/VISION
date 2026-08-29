#!/usr/bin/env python3
"""
Level 2 - The holodeck: a "what-if" multiverse simulator.

Behind the name is a structural causal model (SCM): a DAG where each node is a
function of its parents plus its own noise. That structure buys three things a
backtest cannot give you:

  observation     - what usually happens              p(y)
  intervention    - what happens if we FORCE x        p(y | do(x))
  counterfactual  - what would have happened in THIS
                    exact world had we acted otherwise (twin-network: reuse the
                    same noise draws, change the assignment)

The third is the one that matters. "Would this trade have worked in the world
that actually happened?" is a different question from "does it work on average",
and only the counterfactual answers it.

A "branch" is one intervention rolled forward. A "multiverse" is the set of
branches compared against the baseline. Regret is the honest scoreboard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

# A node's assignment: f(parent_values, noise) -> value array
Assignment = Callable[[Mapping[str, np.ndarray], np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Node:
    name: str
    parents: tuple[str, ...]
    fn: Assignment
    noise_scale: float = 1.0


@dataclass
class World:
    """One rollout: every node's values plus the exact noise that produced them."""

    values: dict[str, np.ndarray]
    noise: dict[str, np.ndarray]
    label: str = "observational"

    def __getitem__(self, name: str) -> np.ndarray:
        return self.values[name]

    def mean(self, name: str) -> float:
        return float(np.mean(self.values[name]))


@dataclass
class Branch:
    """One intervened world, scored against the baseline it forked from."""

    label: str
    interventions: dict[str, float]
    world: World
    outcome: str

    @property
    def outcome_mean(self) -> float:
        return self.world.mean(self.outcome)

    def regret_vs(self, baseline: World) -> float:
        """How much better the baseline did. Positive regret = this branch was worse."""
        return float(baseline.mean(self.outcome) - self.outcome_mean)


class CausalGraph:
    """A DAG of structural assignments. Immutable once you start sampling."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}

    def add(self, name: str, parents: Sequence[str], fn: Assignment, noise_scale: float = 1.0) -> "CausalGraph":
        for p in parents:
            if p not in self._nodes:
                raise ValueError(f"'{name}' declares unknown parent '{p}' - add parents first.")
        if name in self._nodes:
            raise ValueError(f"node '{name}' already exists")
        self._nodes[name] = Node(name, tuple(parents), fn, noise_scale)
        return self

    @property
    def names(self) -> list[str]:
        return list(self._nodes)

    def order(self) -> list[str]:
        """Topological order. Insertion already enforces it, so this just confirms."""
        seen: list[str] = []
        for name, node in self._nodes.items():
            if any(p not in seen for p in node.parents):
                raise ValueError(f"'{name}' precedes a parent - the graph is not a DAG")
            seen.append(name)
        return seen

    def draw_noise(self, n: int, seed: int) -> dict[str, np.ndarray]:
        """Noise is drawn once and kept, so counterfactuals can reuse this world."""
        rng = np.random.default_rng(seed)
        return {name: rng.normal(0, node.noise_scale, n) for name, node in self._nodes.items()}

    def rollout(
        self,
        noise: Mapping[str, np.ndarray],
        interventions: Mapping[str, float] | None = None,
        label: str = "observational",
    ) -> World:
        """Run the graph forward. `interventions` replace a node's assignment with do(x)."""
        do = dict(interventions or {})
        unknown = set(do) - set(self._nodes)
        if unknown:
            raise ValueError(f"cannot intervene on unknown node(s): {sorted(unknown)}")

        values: dict[str, np.ndarray] = {}
        for name in self.order():
            node = self._nodes[name]
            if name in do:
                # do(x) severs the node from its parents entirely - that is the point
                values[name] = np.full_like(noise[name], float(do[name]), dtype=float)
            else:
                values[name] = np.asarray(node.fn(values, noise[name]), dtype=float)
        return World(values, dict(noise), label)

    def sample(self, n: int = 1000, seed: int = 0) -> World:
        return self.rollout(self.draw_noise(n, seed))

    def do(self, n: int = 1000, seed: int = 0, **interventions: float) -> World:
        """p(y | do(x)): a fresh world where we forced x."""
        label = "do(" + ", ".join(f"{k}={v}" for k, v in interventions.items()) + ")"
        return self.rollout(self.draw_noise(n, seed), interventions, label)

    def counterfactual(self, factual: World, **interventions: float) -> World:
        """
        THIS world, had we acted differently.

        Abduction is free here because we kept the noise: re-running the same
        draws with a changed assignment is the twin network.
        """
        label = "counterfactual(" + ", ".join(f"{k}={v}" for k, v in interventions.items()) + ")"
        return self.rollout(factual.noise, interventions, label)


class Multiverse:
    """Fork a baseline world across many interventions and rank the outcomes."""

    def __init__(self, graph: CausalGraph, outcome: str) -> None:
        if outcome not in graph.names:
            raise ValueError(f"outcome '{outcome}' is not a node in the graph")
        self.graph = graph
        self.outcome = outcome
        self.baseline: World | None = None
        self.branches: list[Branch] = []

    def explore(
        self,
        scenarios: Mapping[str, Mapping[str, float]],
        n: int = 2000,
        seed: int = 0,
        counterfactual: bool = True,
    ) -> "Multiverse":
        """
        `scenarios` maps a label -> the interventions that define that branch.

        counterfactual=True forks THIS world (shared noise, branches directly
        comparable). False draws a fresh world per branch (independent futures).
        """
        self.baseline = self.graph.sample(n=n, seed=seed)
        self.branches = []
        for label, interventions in scenarios.items():
            world = (
                self.graph.counterfactual(self.baseline, **interventions)
                if counterfactual
                else self.graph.do(n=n, seed=seed, **interventions)
            )
            self.branches.append(Branch(label, dict(interventions), world, self.outcome))
        return self

    def ranked(self) -> list[Branch]:
        return sorted(self.branches, key=lambda b: b.outcome_mean, reverse=True)

    def report(self) -> str:
        if self.baseline is None:
            return "nothing explored yet"
        base = self.baseline.mean(self.outcome)
        lines = [
            f"outcome '{self.outcome}' | baseline {base:+.4f} | {len(self.branches)} branches",
            "  (all figures SIMULATED - they size nothing, see CLAUDE.md rule 5)",
        ]
        for b in self.ranked():
            delta = b.outcome_mean - base
            lines.append(
                f"  {b.label:<22} {b.outcome_mean:+.4f}  "
                f"({delta:+.4f} vs baseline, regret {b.regret_vs(self.baseline):+.4f})"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    # A toy market: liquidity drives spread; our own size moves the price against us.
    g = (
        CausalGraph()
        .add("liquidity", (), lambda v, e: 1.0 + 0.3 * e, noise_scale=1.0)
        .add("spread_bps", ("liquidity",), lambda v, e: np.maximum(0.5, 6.0 / np.maximum(v["liquidity"], 0.2) + 0.4 * e))
        .add("size", (), lambda v, e: np.full_like(e, 0.05))
        .add("impact_bps", ("size", "liquidity"),
             lambda v, e: 40.0 * v["size"] / np.maximum(v["liquidity"], 0.2) + 0.5 * np.abs(e))
        .add("alpha_bps", (), lambda v, e: 12.0 + 4.0 * e)
        .add("pnl_bps", ("alpha_bps", "spread_bps", "impact_bps"),
             lambda v, e: v["alpha_bps"] - v["spread_bps"] - v["impact_bps"] + 0.2 * e)
    )

    print("STARK MK18 - causal engine demo\n")
    mv = Multiverse(g, outcome="pnl_bps").explore(
        {
            "half size":     {"size": 0.025},
            "as proposed":   {"size": 0.05},
            "double size":   {"size": 0.10},
            "quarter, thin": {"size": 0.025, "liquidity": 0.4},
        },
        n=4000,
        seed=11,
    )
    print(mv.report())

    base = g.sample(n=4000, seed=11)
    cf = g.counterfactual(base, size=0.0)
    print(f"\ncounterfactual: had we not traded at all in THIS world, "
          f"pnl {base.mean('pnl_bps'):+.2f} -> {cf.mean('pnl_bps'):+.2f} bps")
