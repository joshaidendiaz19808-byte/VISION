#!/usr/bin/env python3
"""
Level 2 - Surprise & free-energy minimization.

STARK holds beliefs about the world as Gaussians. When reality arrives, three
things happen, in this order:

  surprise      -log p(observation | belief)   - how wrong were we?
  update        Bayesian conjugate step        - become less wrong
  free energy   complexity - accuracy          - what did that cost us?

Free energy is the honest scorekeeper. A belief that fits the data only by
lurching every time it sees something new pays for that lurch in complexity
(the KL from prior to posterior). Good beliefs explain a lot and move a little.

Then it plans. Expected free energy of an action has two terms:

  pragmatic value  - do we expect this to make money?
  epistemic value  - do we expect to LEARN something?

Minimizing their sum is why STARK will sometimes run a small test it does not
expect to profit from: an experiment that resolves a belief is worth paying for.
Curiosity, budgeted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class Belief:
    """A Gaussian belief about one quantity (e.g. an edge, in bps)."""

    name: str
    mean: float
    variance: float

    @property
    def sd(self) -> float:
        return math.sqrt(max(self.variance, 1e-12))

    @property
    def precision(self) -> float:
        return 1.0 / max(self.variance, 1e-12)

    def surprise(self, observation: float, obs_variance: float = 1.0) -> float:
        """-log p(obs). In nats. Large = reality disagrees with us."""
        var = max(self.variance + obs_variance, 1e-12)
        return float(0.5 * math.log(2 * math.pi * var) + (observation - self.mean) ** 2 / (2 * var))

    def updated(self, observation: float, obs_variance: float = 1.0) -> "Belief":
        """Conjugate Gaussian update. Precision adds; the mean moves by how much we trust the news."""
        p_prior, p_obs = self.precision, 1.0 / max(obs_variance, 1e-12)
        p_post = p_prior + p_obs
        return Belief(self.name, (self.mean * p_prior + observation * p_obs) / p_post, 1.0 / p_post)

    def kl_from(self, other: "Belief") -> float:
        """KL(self || other) for two Gaussians. The 'complexity' cost of moving."""
        v_s, v_o = max(self.variance, 1e-12), max(other.variance, 1e-12)
        return float(0.5 * (math.log(v_o / v_s) + (v_s + (self.mean - other.mean) ** 2) / v_o - 1.0))

    def __str__(self) -> str:
        return f"{self.name}: {self.mean:+.2f} +/- {self.sd:.2f}"


@dataclass(frozen=True)
class Observation:
    belief: str
    value: float
    variance: float = 1.0
    source: str = "live"


@dataclass(frozen=True)
class UpdateResult:
    belief_before: Belief
    belief_after: Belief
    surprise: float
    complexity: float       # KL(posterior || prior) - what the update cost
    accuracy: float         # log p(obs | posterior) - what it bought
    free_energy: float      # complexity - accuracy

    @property
    def shocked(self) -> bool:
        """Surprise past ~3 sigma. The signal that a model, not a parameter, is wrong."""
        return self.surprise > 4.0

    def __str__(self) -> str:
        return (
            f"{self.belief_before} -> {self.belief_after} | "
            f"surprise {self.surprise:.2f} | F {self.free_energy:+.2f}"
            + ("  <- SHOCK: re-examine the model, not the parameter" if self.shocked else "")
        )


@dataclass(frozen=True)
class Action:
    """A candidate experiment: test a hypothesis at some size."""

    name: str
    belief: str
    expected_value: float       # est. bps if we are right
    information_gain: float     # est. reduction in belief variance, [0, 1]
    cost: float = 0.0           # bps to run the experiment


@dataclass
class ActiveInferenceLoop:
    """
    Holds beliefs, absorbs observations, and picks what to do next.

    Deterministic: no sampling anywhere in here.
    """

    beliefs: dict[str, Belief] = field(default_factory=dict)
    history: list[UpdateResult] = field(default_factory=list)
    epistemic_weight: float = 1.0   # raise to buy more curiosity, lower to exploit

    def believe(self, name: str, mean: float, variance: float) -> Belief:
        self.beliefs[name] = Belief(name, float(mean), float(variance))
        return self.beliefs[name]

    def observe(self, obs: Observation) -> UpdateResult:
        prior = self.beliefs.get(obs.belief)
        if prior is None:
            raise KeyError(f"no belief named '{obs.belief}' - call believe() first")
        surprise = prior.surprise(obs.value, obs.variance)
        post = prior.updated(obs.value, obs.variance)
        complexity = post.kl_from(prior)
        accuracy = -0.5 * (
            math.log(2 * math.pi * max(obs.variance, 1e-12))
            + (obs.value - post.mean) ** 2 / max(obs.variance, 1e-12)
        )
        self.beliefs[obs.belief] = post
        result = UpdateResult(prior, post, surprise, complexity, accuracy, complexity - accuracy)
        self.history.append(result)
        return result

    def expected_free_energy(self, action: Action) -> float:
        """
        G(a) = -(pragmatic + epistemic). Lower is better, so we minimize.

        Epistemic value is scaled by how uncertain we currently are: learning
        about something we already know is worth nothing.
        """
        belief = self.beliefs.get(action.belief)
        uncertainty = belief.sd if belief else 1.0
        pragmatic = action.expected_value - action.cost
        epistemic = self.epistemic_weight * action.information_gain * uncertainty
        return float(-(pragmatic + epistemic))

    def select_action(self, actions: Iterable[Action]) -> tuple[Action, dict[str, float]]:
        scored = {a.name: self.expected_free_energy(a) for a in actions}
        if not scored:
            raise ValueError("no actions to choose from")
        best = min(scored, key=scored.get)
        return next(a for a in actions if a.name == best), scored

    def total_free_energy(self) -> float:
        return float(sum(h.free_energy for h in self.history))

    def summary(self) -> str:
        lines = ["beliefs:"] + [f"  {b}" for b in self.beliefs.values()]
        if self.history:
            shocks = sum(h.shocked for h in self.history)
            lines += [
                f"observations: {len(self.history)} | cumulative F {self.total_free_energy():+.2f}"
                + (f" | {shocks} shock(s)" if shocks else "")
            ]
        return "\n".join(lines)


if __name__ == "__main__":
    print("STARK MK18 - active inference demo\n")
    loop = ActiveInferenceLoop()
    loop.believe("momentum_edge_bps", mean=15.0, variance=25.0)   # optimistic, uncertain
    loop.believe("meanrev_edge_bps", mean=2.0, variance=1.0)      # modest, fairly sure

    print("Reality arrives on the momentum book (truth is nearer 4 bps):")
    for obs in [3.5, 5.0, 2.0, 4.5, -18.0]:
        r = loop.observe(Observation("momentum_edge_bps", obs, variance=9.0))
        print(f"  saw {obs:+6.1f}  {r}")

    print("\n" + loop.summary())

    print("\nWhat should we run next?")
    actions = [
        Action("size up momentum",    "momentum_edge_bps", expected_value=4.0,  information_gain=0.10, cost=1.0),
        Action("small momentum probe","momentum_edge_bps", expected_value=0.4,  information_gain=0.80, cost=0.5),
        Action("size up mean-rev",    "meanrev_edge_bps",  expected_value=2.0,  information_gain=0.05, cost=0.5),
        Action("do nothing",          "meanrev_edge_bps",  expected_value=0.0,  information_gain=0.00, cost=0.0),
    ]
    def show(loop: ActiveInferenceLoop) -> None:
        chosen, scores = loop.select_action(actions)
        for name, g in sorted(scores.items(), key=lambda kv: kv[1]):
            print(f"  G={g:+.2f}  {name}")
        a = chosen
        sd = loop.beliefs[a.belief].sd
        prag = a.expected_value - a.cost
        epis = loop.epistemic_weight * a.information_gain * sd
        print(f"  => {a.name}  (pragmatic {prag:+.2f}, epistemic {epis:+.2f})\n")

    print(f"curiosity dialled to {loop.epistemic_weight:.1f} - P&L leads:")
    show(loop)

    loop.epistemic_weight = 6.0
    print(f"curiosity dialled to {loop.epistemic_weight:.1f} - STARK pays to learn instead:")
    show(loop)
    print("Same beliefs, same actions. Only the price of information changed.")
