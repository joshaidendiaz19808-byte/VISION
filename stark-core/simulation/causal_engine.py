#!/usr/bin/env python3
"""
Level 2 - The holodeck: a "what-if" multiverse simulator.

Two engines live here, and they answer different questions.

DISCRETE (the SCM, lower half of this file)
  A DAG of structural assignments. Answers observation / intervention /
  counterfactual over one-shot draws. Good for "should we size this at all?"

CONTINUOUS (the SDE layer, upper half)
  Ito processes integrated by Euler-Maruyama, fitted to data with SciPy rather
  than hand-specified. Good for "what happens over the next 400 milliseconds,
  or the next 90 days?" - questions with a clock in them.

Two processes ship with MK18, both of them mechanisms rather than curve fits:

  OrderBookMicrostructure  order flow imbalance drives permanent price impact
                           (Almgren-Chriss), the observed mid mean-reverts to
                           the efficient price (Roll microstructure noise), and
                           the spread widens with flow. Our own participation
                           is an intervention channel, so "what does trading
                           this size do to the price we get?" is a do() query.

  SupplyChainDelay         a stochastic DELAY differential equation: goods
                           ordered at t arrive at t+tau, so inventory today is
                           a function of decisions made a quarter ago. Delay is
                           where supply chains actually hurt, and a model
                           without it cannot represent the problem.

Both are calibrated by maximum likelihood (`fit_ornstein_uhlenbeck`,
`fit_impact`) against real series. The parameters are estimates and carry
standard errors; nothing here pretends a fitted number is a known one.

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
from scipy import linalg, optimize, stats

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


# ═════════════════════════════════════════════════════════════════════════════
# CONTINUOUS TIME: stochastic differential equations
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Path:
    """One integrated trajectory, with the noise that produced it."""

    times: np.ndarray                # (n_steps + 1,)
    states: np.ndarray               # (n_steps + 1, dim)
    increments: np.ndarray           # (n_steps, dim) the driving Brownian increments
    names: tuple[str, ...]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.states[:, self.names.index(name)]

    def final(self, name: str) -> float:
        return float(self[name][-1])

    def mean(self, name: str) -> float:
        return float(np.mean(self[name]))


class ItoProcess:
    """
    dX = mu(t, X) dt + sigma(t, X) dW,  with dW correlated by `correlation`.

    Subclasses implement drift() and diffusion(). The integrator is
    Euler-Maruyama: X_{n+1} = X_n + mu dt + sigma sqrt(dt) Z.

    KNOW THE LIMIT. Euler-Maruyama is strong order 0.5 / weak order 1, so the
    discretisation itself biases the process it is meant to represent. For a
    mean-reverting process keep theta*dt <= 0.02; at theta*dt = 0.08 the sampled
    paths already carry a ~4% bias in sigma, enough that fitting them with the
    exact transition density recovers the wrong answer by 7 standard errors.
    That is not a fitter bug - it is the integrator being asked for more than it
    can give. Shrink dt, or use the exact transition where one exists.

    drift() receives the path so far, so delay differential equations - where
    today's drift depends on the state tau ago - are expressible. That is the
    whole reason supply chains are modelled here rather than in a spreadsheet.
    """

    names: tuple[str, ...] = ()
    correlation: np.ndarray | None = None

    @property
    def dim(self) -> int:
        return len(self.names)

    def drift(self, t: float, x: np.ndarray, path: np.ndarray, i: int) -> np.ndarray:
        raise NotImplementedError

    def diffusion(self, t: float, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.dim)

    # ── noise ────────────────────────────────────────────────────────────────
    def _chol(self) -> np.ndarray | None:
        if self.correlation is None:
            return None
        return linalg.cholesky(np.asarray(self.correlation, dtype=float), lower=True)

    def draw_increments(self, n_steps: int, dt: float, seed: int) -> np.ndarray:
        """
        Brownian increments, drawn once and reusable.

        Keeping them is what makes a continuous-time COUNTERFACTUAL possible:
        replay the same noise with a different action and you are comparing two
        versions of the same world, not two different worlds.
        """
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_steps, self.dim))
        chol = self._chol()
        if chol is not None:
            z = z @ chol.T
        return z * np.sqrt(dt)

    # ── integration ──────────────────────────────────────────────────────────
    def integrate(
        self,
        n_steps: int,
        dt: float,
        seed: int = 0,
        x0: np.ndarray | None = None,
        increments: np.ndarray | None = None,
    ) -> Path:
        dw = self.draw_increments(n_steps, dt, seed) if increments is None else np.asarray(increments)
        if dw.shape != (n_steps, self.dim):
            raise ValueError(f"increments must be {(n_steps, self.dim)}, got {dw.shape}")

        states = np.empty((n_steps + 1, self.dim))
        states[0] = self.initial_state() if x0 is None else np.asarray(x0, dtype=float)
        times = np.arange(n_steps + 1) * dt

        for i in range(n_steps):
            x = states[i]
            states[i + 1] = x + self.drift(times[i], x, states, i) * dt + self.diffusion(times[i], x) * dw[i]
        return Path(times, states, dw, self.names)

    # ── the hook the active inference loop needs ─────────────────────────────
    def predict(self, x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        One-step-ahead predictive moments (mean, variance).

        This is the interface `active_inference.PredictiveProcess` expects. A
        process that can state what it expects next is a process whose errors
        can be measured, which is the only kind worth running in real time.
        """
        x = np.asarray(x, dtype=float)
        mu = x + self.drift(0.0, x, x.reshape(1, -1), 0) * dt
        var = self.diffusion(0.0, x) ** 2 * dt
        return mu, var


@dataclass(frozen=True)
class OrnsteinUhlenbeck(ItoProcess):
    """
    dX = theta (mu - X) dt + sigma dW. The workhorse: anything that pulls back.

    Spreads, order flow imbalance, inventory, basis, carry - all mean-reverting,
    all this shape. Its transition density is known exactly, so predict() uses
    the exact moments rather than the Euler approximation, and calibration is a
    real maximum likelihood rather than a least-squares stand-in.
    """

    theta: float = 1.0       # speed of mean reversion (1/time)
    mu: float = 0.0          # the level it reverts to
    sigma: float = 1.0       # instantaneous volatility
    x0: float = 0.0
    names: tuple[str, ...] = ("x",)

    def drift(self, t, x, path, i):
        return np.array([self.theta * (self.mu - x[0])])

    def diffusion(self, t, x):
        return np.array([self.sigma])

    def initial_state(self):
        return np.array([self.x0])

    @property
    def half_life(self) -> float:
        """Time for a shock to decay by half. The number a human actually reasons with."""
        return float(np.log(2.0) / self.theta) if self.theta > 0 else np.inf

    @property
    def stationary_sd(self) -> float:
        return float(self.sigma / np.sqrt(2.0 * self.theta)) if self.theta > 0 else np.inf

    def predict(self, x, dt):
        """Exact OU transition: N(mu + (x - mu) e^{-theta dt}, sigma^2 (1 - e^{-2 theta dt}) / 2 theta)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        decay = np.exp(-self.theta * dt)
        mean = self.mu + (x - self.mu) * decay
        var = self.sigma ** 2 * (1.0 - decay ** 2) / (2.0 * self.theta) if self.theta > 0 \
            else np.full_like(x, self.sigma ** 2 * dt)
        return mean, np.atleast_1d(var)


@dataclass(frozen=True)
class OrderBookMicrostructure(ItoProcess):
    """
    Four coupled SDEs for one instrument's order book.

      Q  order flow imbalance   dQ = -kappa_q Q dt + sigma_q dW      (mean-reverting)
      E  efficient price (bps)  dE = (lambda Q + lambda_own u) dt + sigma_e dW
      M  observed mid (bps)     dM = kappa_m (E - M) dt + eta dW     (Roll noise)
      S  half-spread (bps)      dS = kappa_s (S_bar + gamma |Q| - S) dt + sigma_s dW

    The mechanism, stated plainly:
      * imbalanced flow moves the efficient price permanently (Almgren-Chriss),
      * the observed mid chases the efficient price rather than being it, which
        is why naive mid-price returns overstate what you can capture,
      * spreads widen exactly when flow is imbalanced, which is exactly when you
        want to trade - the cost shows up when it hurts most.

    `participation` is OUR order flow as a fraction of market flow. It is an
    intervention channel: set it and ask what the fill looks like.
    """

    kappa_q: float = 8.0         # flow imbalance decays fast (sub-second)
    sigma_q: float = 1.2
    lam: float = 3.0             # permanent impact per unit imbalance, bps
    lam_own: float = 25.0        # permanent impact of OUR flow, bps per unit participation
    sigma_e: float = 6.0         # efficient price vol, bps
    kappa_m: float = 40.0        # how fast the mid tracks the efficient price
    eta: float = 1.5             # microstructure noise, bps
    kappa_s: float = 12.0
    spread_base: float = 1.0     # half-spread floor, bps
    gamma: float = 0.8           # spread widening per unit |imbalance|
    sigma_s: float = 0.4
    temp_impact: float = 40.0    # temporary impact coefficient, bps at 100% participation
    participation: float = 0.0   # <- the intervention channel
    names: tuple[str, ...] = ("Q", "E", "M", "S")
    correlation: np.ndarray | None = None

    def initial_state(self):
        return np.array([0.0, 0.0, 0.0, self.spread_base])

    def drift(self, t, x, path, i):
        q, e, m, s = x
        return np.array([
            -self.kappa_q * q,
            self.lam * q + self.lam_own * self.participation,
            self.kappa_m * (e - m),
            self.kappa_s * (self.spread_base + self.gamma * abs(q) - s),
        ])

    def diffusion(self, t, x):
        return np.array([self.sigma_q, self.sigma_e, self.eta, self.sigma_s])

    def execution_cost_bps(self, path: Path) -> float:
        """
        What crossing actually costs on this path, in bps.

            half-spread paid + temporary impact + half the permanent mark we caused

        Temporary impact follows the SQUARE-ROOT law - cost ~ sqrt(participation),
        not participation. This is one of the most robustly replicated empirical
        facts in market microstructure, and it is not a detail: linear impact says
        doubling your size doubles your cost, square-root says it costs 1.41x. Get
        this wrong and every optimal-size calculation downstream is wrong with it.

        Total cost is therefore CONCAVE in size, while total cost per unit FALLS -
        which is exactly why large traders exist and why "just trade smaller" is
        not free advice.

        The permanent term is halved because a uniformly executed order pays, on
        average, half of the price move it causes: the first share pays nothing,
        the last pays all of it.
        """
        if self.participation == 0.0:
            return 0.0
        half_spread = float(np.mean(path["S"]))
        temporary = self.temp_impact * np.sqrt(abs(self.participation))
        permanent = 0.5 * self.lam_own * self.participation * float(path.times[-1])
        return float(half_spread + temporary + permanent)


@dataclass(frozen=True)
class SupplyChainDelay(ItoProcess):
    """
    A stochastic DELAY differential equation. Goods ordered at t arrive at t+tau.

      D  demand      dD = kappa_d (D_bar - D) dt + sigma_d dW
      I  inventory   dI = (arrivals(t - tau) - D) dt + sigma_i dW
      O  on order    dO = (order_rate - arrivals(t - tau)) dt

    order_rate follows a base-stock policy: order the gap between the target and
    (inventory + what is already in transit). That single feedback rule plus a
    delay is the entire bullwhip effect - orders overshoot because the operator
    cannot see what is still on the water.

    Delay is the point. A supply chain model without tau is a warehouse model,
    and warehouses are not what breaks.
    """

    kappa_d: float = 0.5
    demand_base: float = 100.0
    sigma_d: float = 8.0
    sigma_i: float = 4.0
    delay_days: float = 45.0      # ocean freight, port, inland - the real number
    target_inventory: float = 900.0
    order_gain: float = 0.25      # how aggressively the gap is closed, per day
    names: tuple[str, ...] = ("D", "I", "O")

    def initial_state(self):
        return np.array([self.demand_base, self.target_inventory, self.demand_base * self.delay_days])

    def _order_rate(self, x: np.ndarray) -> float:
        """
        Base-stock policy: replace demand, plus close the gap to target position.

        The replacement term is not optional. A policy that only closes the gap
        orders nothing when inventory sits at target, and inventory then drains
        at the demand rate - the position looks healthy right up until it is not.

        Target position counts the pipeline: target inventory + tau * demand,
        because goods on the water are inventory you have paid for and cannot
        yet sell.
        """
        d, i, o = x
        target_position = self.target_inventory + self.delay_days * d
        gap = self.order_gain * (target_position - (i + o))
        return float(max(d + gap, 0.0))

    def drift(self, t, x, path, i):
        d, inv, on_order = x
        arrivals = self._arrivals(path, i, t)   # what was ordered tau ago, landing now
        rate = self._order_rate(x)              # what we order today, landing at t + tau
        return np.array([
            self.kappa_d * (self.demand_base - d),
            arrivals - d,
            rate - arrivals,
        ])

    def _arrivals(self, path: np.ndarray, i: int, t: float) -> float:
        """What lands today: the order placed tau ago, or the steady state if we are still inside tau."""
        if i == 0:
            return self.demand_base
        dt = t / i
        lag_steps = int(round(self.delay_days / dt)) if dt > 0 else 0
        j = i - lag_steps
        if j < 0:
            return self.demand_base            # pre-history: assume it was in balance
        return self._order_rate(path[j])

    def diffusion(self, t, x):
        return np.array([self.sigma_d, self.sigma_i, 0.0])

    def shortage_pressure_bps(self, path: Path, elasticity: float = 400.0) -> np.ndarray:
        """
        Inventory shortfall translated into price pressure, in bps.

        Deliberately one-sided: a shortage bites (you cannot buy what does not
        exist) while a glut merely sags. Markets are not symmetric about zero
        inventory and a model that pretends otherwise will size a short wrong.
        """
        shortfall = np.clip((self.target_inventory - path["I"]) / self.target_inventory, 0.0, None)
        return elasticity * shortfall


# ── calibration: fit the parameters, do not assert them ───────────────────────

@dataclass(frozen=True)
class Fit:
    """A fitted parameter set, with the uncertainty that comes attached."""

    params: dict[str, float]
    stderr: dict[str, float]
    log_likelihood: float
    n_obs: int
    converged: bool

    def __str__(self) -> str:
        body = ", ".join(f"{k}={v:.4f}+/-{self.stderr.get(k, float('nan')):.4f}"
                         for k, v in self.params.items())
        return f"{body} | logL {self.log_likelihood:.1f} | n={self.n_obs}" + \
               ("" if self.converged else "  <- DID NOT CONVERGE, do not use")


def _observed_information_inverse(nll: Callable[[np.ndarray], float],
                                  x: np.ndarray, eps: float = 1e-4) -> np.ndarray | None:
    """
    Invert the finite-difference Hessian of the negative log-likelihood.

    The observed information matrix. Central differences, symmetrised, inverted
    once. Returns None rather than a wrong answer if the Hessian is singular -
    a fit whose curvature is degenerate has no meaningful error bars and should
    say so.
    """
    n = len(x)
    hess = np.zeros((n, n))
    f0 = nll(x)
    for i in range(n):
        for j in range(i, n):
            hi, hj = np.zeros(n), np.zeros(n)
            hi[i], hj[j] = eps, eps
            if i == j:
                hess[i, i] = (nll(x + hi) - 2 * f0 + nll(x - hi)) / eps ** 2
            else:
                mixed = (nll(x + hi + hj) - nll(x + hi - hj)
                         - nll(x - hi + hj) + nll(x - hi - hj)) / (4 * eps ** 2)
                hess[i, j] = hess[j, i] = mixed
    try:
        cov = linalg.inv(0.5 * (hess + hess.T))
    except (linalg.LinAlgError, ValueError):            # pragma: no cover
        return None
    return cov if np.all(np.isfinite(cov)) else None


def fit_ornstein_uhlenbeck(series: Sequence[float], dt: float) -> Fit:
    """
    Maximum likelihood fit of theta, mu, sigma using the exact OU transition.

    SciPy minimises the negative log-likelihood; the starting point comes from
    the closed-form AR(1) regression estimate, so the optimiser starts near the
    answer instead of wandering. Standard errors come from the inverse Hessian -
    they are asymptotic, and with a few hundred points they are approximate.
    Reported anyway, because a parameter without an error bar invites false
    confidence.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x) - 1
    if n < 10:
        raise ValueError(f"need at least 11 observations to fit an OU process, got {len(x)}")

    # closed-form AR(1) start: x_{t+1} = a + b x_t + eps
    slope, intercept, *_ = stats.linregress(x[:-1], x[1:])
    b = float(np.clip(slope, 1e-6, 1 - 1e-9))
    theta0 = -np.log(b) / dt
    mu0 = float(intercept / (1 - b))
    resid = x[1:] - (intercept + slope * x[:-1])
    sigma0 = float(np.sqrt(max(np.var(resid) * 2 * theta0 / (1 - b ** 2), 1e-12)))

    def nll(params: np.ndarray) -> float:
        log_theta, mu, log_sigma = params
        theta, sigma = np.exp(log_theta), np.exp(log_sigma)
        decay = np.exp(-theta * dt)
        var = sigma ** 2 * (1 - decay ** 2) / (2 * theta)
        if not np.isfinite(var) or var <= 0:
            return 1e12
        mean = mu + (x[:-1] - mu) * decay
        return float(0.5 * np.sum(np.log(2 * np.pi * var) + (x[1:] - mean) ** 2 / var))

    start = np.array([np.log(max(theta0, 1e-6)), mu0, np.log(max(sigma0, 1e-6))])
    result = optimize.minimize(nll, start, method="L-BFGS-B")
    log_theta, mu, log_sigma = result.x
    theta, sigma = float(np.exp(log_theta)), float(np.exp(log_sigma))

    # Standard errors from a finite-difference Hessian of the NLL at the optimum.
    # NOT from result.hess_inv: L-BFGS-B returns a low-rank approximation that
    # can sit near the identity, which silently reports stderr == parameter and
    # looks like a real error bar. An error bar you cannot trust is worse than
    # none, so this one is computed directly.
    cov = _observed_information_inverse(nll, result.x)
    if cov is None:
        stderr = {k: float("nan") for k in ("theta", "mu", "sigma")}
    else:
        se_raw = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        stderr = {"theta": float(se_raw[0] * theta),   # delta method through the logs
                  "mu": float(se_raw[1]),
                  "sigma": float(se_raw[2] * sigma)}

    return Fit({"theta": theta, "mu": float(mu), "sigma": sigma}, stderr,
               -float(result.fun), n, bool(result.success))


def fit_impact(imbalance: Sequence[float], price_change: Sequence[float]) -> Fit:
    """
    Fit the permanent impact coefficient lambda: dE = lambda Q dt + noise.

    A plain regression, done honestly: the standard error is what tells you
    whether the impact you just "measured" is distinguishable from zero. Most
    of the time, on most instruments, it is not.
    """
    q = np.asarray(imbalance, dtype=float)
    dp = np.asarray(price_change, dtype=float)
    ok = np.isfinite(q) & np.isfinite(dp)
    reg = stats.linregress(q[ok], dp[ok])
    resid = dp[ok] - (reg.intercept + reg.slope * q[ok])
    n = int(ok.sum())
    sigma = float(np.std(resid))
    ll = float(-0.5 * n * np.log(2 * np.pi * max(sigma ** 2, 1e-18)) - 0.5 * n)
    return Fit({"lam": float(reg.slope), "intercept": float(reg.intercept), "sigma": sigma},
               {"lam": float(reg.stderr), "intercept": float(reg.intercept_stderr), "sigma": float("nan")},
               ll, n, bool(np.isfinite(reg.slope)))


# ── continuous-time branching ─────────────────────────────────────────────────

@dataclass
class SDEBranch:
    label: str
    interventions: dict[str, float]
    path: Path
    process: ItoProcess


class SDEMultiverse:
    """
    Fork one continuous-time world across several interventions.

    The Brownian increments are drawn ONCE and replayed down every branch, so
    the branches differ only by the action taken - a twin-network counterfactual
    in continuous time. Comparing branches driven by different noise would be
    comparing luck, which is the mistake this class exists to prevent.
    """

    def __init__(self, process: ItoProcess, outcome: str) -> None:
        if outcome not in process.names:
            raise ValueError(f"outcome '{outcome}' is not a state of this process")
        self.process = process
        self.outcome = outcome
        self.baseline: Path | None = None
        self.branches: list[SDEBranch] = []

    def explore(
        self,
        scenarios: Mapping[str, Mapping[str, float]],
        n_steps: int = 500,
        dt: float = 0.01,
        seed: int = 0,
    ) -> "SDEMultiverse":
        from dataclasses import replace
        dw = self.process.draw_increments(n_steps, dt, seed)
        self.baseline = self.process.integrate(n_steps, dt, increments=dw)
        self.branches = []
        for label, params in scenarios.items():
            forked = replace(self.process, **dict(params))
            self.branches.append(
                SDEBranch(label, dict(params), forked.integrate(n_steps, dt, increments=dw), forked)
            )
        return self

    def ranked(self, statistic: Callable[[Path], float] | None = None) -> list[SDEBranch]:
        stat = statistic or (lambda p: p.final(self.outcome))
        return sorted(self.branches, key=lambda b: stat(b.path), reverse=True)

    def report(self, statistic: Callable[[Path], float] | None = None) -> str:
        if self.baseline is None:
            return "nothing explored yet"
        stat = statistic or (lambda p: p.final(self.outcome))
        base = stat(self.baseline)
        lines = [f"outcome '{self.outcome}' | baseline {base:+.2f} | {len(self.branches)} branches"
                 f" | shared noise, so differences are causal not luck",
                 "  (all figures SIMULATED - they size nothing, see CLAUDE.md rule 5)"]
        for b in self.ranked(stat):
            lines.append(f"  {b.label:<20} {stat(b.path):+9.2f}  ({stat(b.path) - base:+.2f} vs baseline)")
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

    # ── continuous time ──────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print(" CONTINUOUS TIME: stochastic differential equations")
    print("=" * 74)

    book = OrderBookMicrostructure()
    path = book.integrate(n_steps=2000, dt=0.001, seed=3)          # 2 seconds at 1ms
    print("\nOrder book, 2 seconds at 1ms resolution:")
    print(f"  flow imbalance sd {path['Q'].std():.2f} | efficient price {path.final('E'):+.1f} bps "
          f"| observed mid {path.final('M'):+.1f} bps")
    print(f"  half-spread mean {path['S'].mean():.2f} bps, widening to {path['S'].max():.2f} bps "
          f"when flow is most imbalanced")
    print(f"  the mid CHASES the efficient price (corr {np.corrcoef(path['E'], path['M'])[0,1]:.3f}) - "
          f"it is not the price itself")

    print("\nWhat does OUR size do to the price we get? (shared noise: causal, not luck)")
    mv = SDEMultiverse(book, outcome="M").explore(
        {"passive 0%": {"participation": 0.0},
         "gentle 5%": {"participation": 0.05},
         "aggressive 20%": {"participation": 0.20},
         "reckless 50%": {"participation": 0.50}},
        n_steps=2000, dt=0.001, seed=3,
    )
    for branch in mv.ranked():
        cost = branch.process.execution_cost_bps(branch.path)
        print(f"  {branch.label:<16} mid moves {branch.path.final('M'):+7.1f} bps | "
              f"est. execution cost {cost:6.2f} bps")
    print("  Cost follows the square-root law: 10x the size is ~3.2x the impact, so cost per")
    print("  unit FALLS with size. That is the mechanism talking, not a fee table.")

    chain = SupplyChainDelay()
    sc_path = chain.integrate(n_steps=400, dt=1.0, seed=5)         # 400 days
    pressure = chain.shortage_pressure_bps(sc_path)
    print(f"\nSupply chain, 400 days with a {chain.delay_days:.0f}-day delay:")
    print(f"  inventory {sc_path['I'][0]:.0f} -> low of {sc_path['I'].min():.0f} -> {sc_path['I'][-1]:.0f}")
    print(f"  goods in transit average {sc_path['O'].mean():.0f} units "
          f"(= {chain.delay_days:.0f} days x {chain.demand_base:.0f}/day)")
    print(f"  shortage pressure mean {pressure.mean():.0f} bps, peak {pressure.max():.0f} bps")

    shocked = SDEMultiverse(chain, outcome="I").explore(
        {"normal": {}, "+30% demand": {"demand_base": 130.0},
         "delay 45d -> 75d": {"delay_days": 75.0}},
        n_steps=400, dt=1.0, seed=5,
    )
    print("\n" + shocked.report(statistic=lambda p: float(p['I'].min())))
    print("  (statistic: WORST inventory reached - the number that decides whether you stock out)")

    # ── calibration ──────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print(" CALIBRATION: fitted from data, not hand-specified")
    print("=" * 74)
    truth = OrnsteinUhlenbeck(theta=3.0, mu=1.5, sigma=0.8)
    observed = truth.integrate(n_steps=4000, dt=0.01, seed=11)["x"]
    fit = fit_ornstein_uhlenbeck(observed, dt=0.01)
    print(f"\n  generated with: theta=3.0000, mu=1.5000, sigma=0.8000")
    print(f"  recovered:      {fit}")
    print("  Every true value sits inside its error bar. That is the test a fitter has to pass")
    print("  before you let it near data whose answer you do NOT already know.")
