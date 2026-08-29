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

REAL TIME (PredictionErrorLoop, lower half of this file)
  The loop above learns from whatever you hand it. The loop below learns from a
  running process, one observation at a time: predict the next value, see it,
  measure the error, correct. That is the difference between a backtest and a
  filter.

  A backtest fits parameters on history and freezes them. It cannot notice that
  the world moved, because nothing in it is looking. A filter carries an
  explicit belief about its own bias and revises that belief every step, so a
  regime change shows up as a run of surprise and gets absorbed - in seconds,
  not at the next refit.

  The process being tracked only has to satisfy `PredictiveProcess`: it must be
  able to say what it expects next and how sure it is. Anything that cannot
  state a predictive distribution cannot have its errors measured, and anything
  whose errors cannot be measured has no business allocating capital.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol, Sequence, runtime_checkable

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


# ═════════════════════════════════════════════════════════════════════════════
# REAL TIME: prediction error minimization on a running process
# ═════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class PredictiveProcess(Protocol):
    """
    Anything that can state what it expects next.

    Structural typing on purpose: this file must not import the simulation
    engine (they are the same level - see CLAUDE.md rule 1), and it does not
    need to. A process satisfies this contract by having the method, not by
    inheriting anything. An SDE, a fitted regression, a lookup table and a
    constant all qualify, and the loop treats them identically.
    """

    def predict(self, x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, variance) of the next observation given the current state."""
        ...


@dataclass(frozen=True)
class Step:
    """One predict-observe-correct cycle."""

    t: float
    predicted: float
    observed: float
    variance: float
    error: float
    surprise: float
    bias: float

    @property
    def z(self) -> float:
        """Standardised error. If the model's variance is honest, these are ~N(0,1)."""
        return self.error / math.sqrt(max(self.variance, 1e-12))


@dataclass
class OnlineResult:
    """What a run of the filter actually established."""

    steps: list[Step] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.steps)

    @property
    def rmse(self) -> float:
        if not self.steps:
            return float("nan")
        return float(np.sqrt(np.mean([s.error ** 2 for s in self.steps])))

    @property
    def mean_surprise(self) -> float:
        return float(np.mean([s.surprise for s in self.steps])) if self.steps else float("nan")

    @property
    def cumulative_free_energy(self) -> float:
        """Total surprise. The quantity the whole architecture exists to reduce."""
        return float(np.sum([s.surprise for s in self.steps])) if self.steps else float("nan")

    @property
    def calibration(self) -> float:
        """
        SD of the standardised errors. 1.0 means the model's stated uncertainty
        is the truth. Above 1: overconfident, it is more wrong than it admits.
        Below 1: underconfident, it is hedging.
        """
        return float(np.std([s.z for s in self.steps])) if self.steps else float("nan")

    @property
    def final_bias(self) -> float:
        return self.steps[-1].bias if self.steps else 0.0

    def drifting(self, window: int = 20, threshold: float = 2.5) -> bool:
        """
        Is the model wrong in one direction lately?

        A run of same-signed errors is the signature of a regime change: noise
        alternates, bias does not. This is what tells STARK the model itself
        needs looking at, rather than just its parameters.
        """
        if self.n < window:
            return False
        recent = [s.error for s in self.steps[-window:]]
        return abs(float(np.mean(recent)) / (float(np.std(recent)) / math.sqrt(window) + 1e-12)) > threshold

    def __str__(self) -> str:
        return (f"{self.n} steps | RMSE {self.rmse:.4f} | mean surprise {self.mean_surprise:.3f} "
                f"| calibration {self.calibration:.2f} | bias {self.final_bias:+.4f}"
                + ("  <- DRIFTING" if self.drifting() else ""))


class PredictionErrorLoop:
    """
    Online filter: predict one step ahead, observe, correct. Repeat forever.

    The correction is a belief, not a refit. The loop holds a Gaussian belief
    over an additive bias term - "how wrong is this model, right now, in which
    direction" - and updates it from every observation using the same conjugate
    step the offline loop uses. The process itself is never mutated, so the
    mechanism stays what it was fitted to be and the adaptation stays visible
    and auditable as a single number.

    `adapt=False` gives you the backtest: a frozen model that predicts and never
    learns. It is here so the two can be compared honestly on the same data
    rather than asserted about.
    """

    def __init__(
        self,
        process: PredictiveProcess,
        dt: float,
        observe_index: int = 0,
        adapt: bool = True,
        bias_variance: float = 1.0,
        learning_variance: float | None = None,
    ) -> None:
        self.process = process
        self.dt = float(dt)
        self.observe_index = int(observe_index)
        self.adapt = bool(adapt)
        self.inference = ActiveInferenceLoop()
        self.inference.believe("bias", 0.0, bias_variance)
        self.learning_variance = learning_variance
        self.result = OnlineResult()
        self._t = 0.0

    @property
    def bias(self) -> Belief:
        return self.inference.beliefs["bias"]

    def step(self, state: Sequence[float] | float, observation: float) -> Step:
        """
        One cycle. Call it as data arrives - this is the real-time entry point.

        `state` is where the process is now; `observation` is what actually
        happened one dt later.
        """
        x = np.atleast_1d(np.asarray(state, dtype=float))
        mean, var = self.process.predict(x, self.dt)
        mean = float(np.atleast_1d(mean)[self.observe_index])
        var = float(max(np.atleast_1d(var)[self.observe_index], 1e-12))

        bias = self.bias.mean if self.adapt else 0.0
        predicted = mean + bias
        error = float(observation) - predicted
        surprise = float(0.5 * math.log(2 * math.pi * var) + error ** 2 / (2 * var))

        if self.adapt:
            # the observation implies this bias: whatever closes the gap
            implied = self.bias.mean + error
            self.inference.observe(Observation("bias", implied,
                                               variance=self.learning_variance or var))

        self._t += self.dt
        step = Step(self._t, predicted, float(observation), var, error, surprise, bias)
        self.result.steps.append(step)
        return step

    def run(self, series: Sequence[float], states: Sequence[Sequence[float]] | None = None) -> OnlineResult:
        """
        Filter a whole series. Equivalent to calling step() as each point lands -
        no lookahead anywhere, which is the property that makes this a filter
        and not a fit.
        """
        y = np.asarray(series, dtype=float)
        if len(y) < 2:
            raise ValueError("need at least two observations to predict anything")
        for i in range(len(y) - 1):
            state = y[i] if states is None else states[i]
            self.step(state, float(y[i + 1]))
        return self.result


def compare_online_vs_static(
    process: PredictiveProcess,
    series: Sequence[float],
    dt: float,
    observe_index: int = 0,
) -> dict[str, OnlineResult]:
    """
    Run the same model over the same data twice: frozen, and adapting.

    This is the experiment behind the claim "minimize prediction error in real
    time instead of running a static backtest". It is cheap to run and it is
    falsifiable - if adaptation does not lower cumulative free energy on YOUR
    data, then on your data it does not help, and you should know that before
    building on it.
    """
    return {
        "static": PredictionErrorLoop(process, dt, observe_index, adapt=False).run(series),
        "online": PredictionErrorLoop(process, dt, observe_index, adapt=True).run(series),
    }


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

    # ── real time: filter vs backtest ────────────────────────────────────────
    print("\n" + "=" * 74)
    print(" REAL TIME: prediction error minimization vs a static backtest")
    print("=" * 74)

    import pathlib as _pl, sys as _sys
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    from simulation.causal_engine import OrnsteinUhlenbeck  # demo only - the loop needs no import

    dt = 0.05
    calm = OrnsteinUhlenbeck(theta=2.0, mu=0.0, sigma=0.5)
    first = calm.integrate(600, dt, seed=4)["x"]
    # the world moves: the level it reverts to shifts from 0 to +3 and nobody tells us
    second = OrnsteinUhlenbeck(theta=2.0, mu=3.0, sigma=0.5, x0=float(first[-1])).integrate(600, dt, seed=5)["x"]
    series = np.concatenate([first, second[1:]])

    print("\n  A model fitted when mu=0. Halfway through, the world moves to mu=3.")
    print("  Same model, same data, run two ways:\n")
    results = compare_online_vs_static(calm, series, dt)
    for name, r in results.items():
        label = "static (backtest)" if name == "static" else "online (filter)"
        print(f"  {label:<20} {r}")

    static, online = results["static"], results["online"]
    print(f"\n  RMSE {static.rmse:.4f} -> {online.rmse:.4f} "
          f"({(1 - online.rmse / static.rmse) * 100:.0f}% lower)")
    print(f"  cumulative free energy {static.cumulative_free_energy:+.0f} -> "
          f"{online.cumulative_free_energy:+.0f} (lower is better)")
    print(f"  the frozen model is still biased at the end ({static.final_bias:+.3f} by construction);")
    print(f"  the filter has absorbed the shift into a bias of {online.final_bias:+.3f}.")
    print("\n  The backtest cannot notice the world moved. Nothing in it is looking.")
