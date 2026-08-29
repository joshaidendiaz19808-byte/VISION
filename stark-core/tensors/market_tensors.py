#!/usr/bin/env python3
"""
Level 3 - Thermodynamic equilibrium fields.

The substrate. Everything above this line reasons about markets; this file holds
what a market *is* at a moment in time.

A MarketTensor is a labelled 3-D array: (asset, feature, time). On top of it sit
five fields borrowed from thermodynamics because the analogy is load-bearing, not
decorative:

  temperature T   realised volatility            - how agitated is it?
  entropy     S   dispersion of the return
                  distribution (Shannon, binned) - how disordered is it?
  energy      U   displacement from the local
                  mean, in sigma                 - how far from rest is it?
  pressure    P   -dU/dt, the restoring force    - which way is it being pushed?
  free energy F = U * (1 - T*S)                  - what displacement is NOT
                                                   explained by ambient chaos?

That last one is the whole point. A price 3 sigma from its mean in a calm, ordered
tape is a real dislocation. The same 3 sigma in a hot, high-entropy tape is just
Tuesday. F separates the two; raw z-scores cannot.

Phase 3 adds two more layers on top of that substrate:

  CapitalField        capital, hypotheses and risk limits as three fields over
                      the same index set, relaxed to thermodynamic equilibrium.
                      Minimising F = U - T*S under hard caps. At high
                      temperature the book spreads out; at low temperature it
                      concentrates on its best idea; the caps bind absolutely at
                      every temperature.

  MarketStateNetwork  a tensor network over discrete future market states. A
                      joint distribution across N variables has d^N
                      configurations - 2^40 is a trillion - and this contracts
                      it in O(N d^2) without ever enumerating one. Marginals,
                      conditionals, the most likely joint state, and exact
                      samples, all from the same contraction.

Dependency rule: this module imports stdlib and numpy ONLY. It is the floor of
the building - it depends on nothing, so nothing can corrupt it (CLAUDE.md 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class FieldState:
    """The thermodynamic reading for one asset at one instant."""

    asset: str
    temperature: float
    entropy: float
    energy: float
    pressure: float
    free_energy: float

    @property
    def disequilibrium(self) -> float:
        """|F|: distance from rest that the ambient chaos does NOT explain."""
        return abs(self.free_energy)

    def verdict(self) -> str:
        if self.disequilibrium < 0.5:
            return "at equilibrium - nothing to trade here"
        direction = "stretched high (pressure is downward)" if self.energy > 0 else \
                    "stretched low (pressure is upward)"
        strength = "strongly" if self.disequilibrium > 1.5 else "mildly"
        return f"{strength} {direction}"

    def __str__(self) -> str:
        return (
            f"{self.asset:<8} T {self.temperature:6.4f} | S {self.entropy:5.2f} | "
            f"U {self.energy:+6.2f} | P {self.pressure:+6.2f} | F {self.free_energy:+6.2f}  "
            f"{self.verdict()}"
        )


class MarketTensor:
    """
    (asset, feature, time) with names attached, so an axis mix-up is an error
    rather than a silently wrong number.
    """

    def __init__(
        self,
        data: np.ndarray,
        assets: Sequence[str],
        features: Sequence[str],
        timestamps: Sequence | None = None,
    ) -> None:
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 3:
            raise ValueError(f"expected (asset, feature, time), got shape {arr.shape}")
        if arr.shape[0] != len(assets) or arr.shape[1] != len(features):
            raise ValueError(
                f"shape {arr.shape} does not match {len(assets)} assets x {len(features)} features"
            )
        self.data = arr
        self.assets = list(assets)
        self.features = list(features)
        self.timestamps = list(timestamps) if timestamps is not None else list(range(arr.shape[2]))

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def from_panels(cls, panels: Mapping[str, Mapping[str, Sequence[float]]]) -> "MarketTensor":
        """From {asset: {feature: series}}. Every series must be the same length."""
        assets = list(panels)
        if not assets:
            raise ValueError("no assets given")
        features = list(panels[assets[0]])
        length = len(panels[assets[0]][features[0]])
        block = np.empty((len(assets), len(features), length))
        for i, a in enumerate(assets):
            for j, f in enumerate(features):
                series = np.asarray(panels[a][f], dtype=float)
                if len(series) != length:
                    raise ValueError(f"{a}/{f} has {len(series)} rows, expected {length}")
                block[i, j] = series
        return cls(block, assets, features)

    # ── access ───────────────────────────────────────────────────────────────
    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    def series(self, asset: str, feature: str) -> np.ndarray:
        return self.data[self.assets.index(asset), self.features.index(feature)]

    def panel(self, feature: str) -> np.ndarray:
        """All assets, one feature: shape (asset, time)."""
        return self.data[:, self.features.index(feature), :]

    def window(self, n: int) -> "MarketTensor":
        return MarketTensor(self.data[:, :, -n:], self.assets, self.features, self.timestamps[-n:])

    # ── the fields ───────────────────────────────────────────────────────────
    def temperature(self, feature: str = "return", window: int = 20) -> np.ndarray:
        """Realised volatility over the trailing window. One number per asset."""
        panel = self.panel(feature)[:, -window:]
        return np.std(panel, axis=1)

    def entropy(self, feature: str = "return", window: int = 60, bins: int = 12) -> np.ndarray:
        """
        Shannon entropy of the binned return distribution, normalised to [0, 1].

        1.0 = every outcome equally likely (maximum disorder, no structure).
        Low = returns clustering, which is where structure lives.
        """
        panel = self.panel(feature)[:, -window:]
        out = np.empty(panel.shape[0])
        for i, row in enumerate(panel):
            finite = row[np.isfinite(row)]
            if finite.size < 2 or np.ptp(finite) < EPS:
                out[i] = 0.0
                continue
            counts, _ = np.histogram(finite, bins=bins)
            p = counts[counts > 0] / counts.sum()
            out[i] = float(-(p * np.log(p)).sum() / np.log(bins))
        return out

    def energy(self, feature: str = "price", window: int = 60) -> np.ndarray:
        """Displacement from the local mean, in local sigma. Signed."""
        panel = self.panel(feature)[:, -window:]
        mu = panel.mean(axis=1)
        sd = panel.std(axis=1)
        return (panel[:, -1] - mu) / np.maximum(sd, EPS)

    def pressure(self, feature: str = "price", window: int = 60, lag: int = 5) -> np.ndarray:
        """
        Restoring force: -dU/dt over `lag` bars.

        Positive pressure = energy is falling = the tape is pushing back toward
        rest. Negative = the stretch is still building.
        """
        now = self.energy(feature, window)
        before = MarketTensor(
            self.data[:, :, : self.shape[2] - lag], self.assets, self.features
        ).energy(feature, window)
        return -(now - before) / max(lag, 1)

    def free_energy(
        self,
        price_feature: str = "price",
        return_feature: str = "return",
        window: int = 60,
        temp_window: int = 20,
    ) -> np.ndarray:
        """
        F = U * (1 - T*S), the displacement left over after ambient chaos.

        Temperature arrives in return units and energy in sigma, so T is
        normalised against the cross-section before use. The discount is clamped
        to [0, 1] so F can shrink a stretch to nothing but can never invert its
        sign - a flipped sign would say "stretched the other way", which is a
        different claim than "this stretch means nothing".

        With a single asset there is no cross-section to normalise against, so T
        contributes nothing and F reduces to raw displacement. Stated here rather
        than hidden: one asset is not enough to know what "hot" means.
        """
        u = self.energy(price_feature, window)
        t = self.temperature(return_feature, temp_window)
        s = self.entropy(return_feature, window)
        t_scaled = t / max(float(np.mean(t)), EPS) if len(t) > 1 else np.zeros_like(t)
        discount = np.clip(t_scaled * s, 0.0, 1.0)
        return u * (1.0 - discount)

    # ── readout ──────────────────────────────────────────────────────────────
    def field_state(
        self,
        price_feature: str = "price",
        return_feature: str = "return",
        window: int = 60,
    ) -> list[FieldState]:
        t = self.temperature(return_feature)
        s = self.entropy(return_feature, window)
        u = self.energy(price_feature, window)
        p = self.pressure(price_feature, window)
        f = self.free_energy(price_feature, return_feature, window)
        return [
            FieldState(a, float(t[i]), float(s[i]), float(u[i]), float(p[i]), float(f[i]))
            for i, a in enumerate(self.assets)
        ]

    def disequilibrium_ranking(self, **kw) -> list[FieldState]:
        """Most dislocated first. This is the queue the generator should look at."""
        return sorted(self.field_state(**kw), key=lambda s: s.disequilibrium, reverse=True)

    def report(self, **kw) -> str:
        lines = [f"MarketTensor {self.shape[0]} assets x {self.shape[1]} features x {self.shape[2]} bars",
                 "  T=temperature  S=entropy  U=energy(sigma)  P=pressure  F=U(1-TS)"]
        lines += ["  " + str(s) for s in self.disequilibrium_ranking(**kw)]
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3a: CAPITAL FIELDS AT THERMODYNAMIC EQUILIBRIUM
# ═════════════════════════════════════════════════════════════════════════════

def project_to_capped_simplex(v: np.ndarray, caps: np.ndarray, total: float = 1.0) -> np.ndarray:
    """
    Nearest point to v satisfying: w >= 0, w <= caps, sum(w) == total.

    Found by bisection on a single threshold tau, because
    sum(clip(v - tau, 0, caps)) is monotonically decreasing in tau - so there is
    exactly one tau that hits the budget, and 60 halvings find it to machine
    precision.

    PROJECTION, not a penalty. A penalty lets a constraint be bought off if the
    objective wants it badly enough, which is precisely the behaviour the risk
    arbiter exists to forbid. Here the feasible set is the only place the
    optimiser is allowed to stand.
    """
    v = np.asarray(v, dtype=float)
    caps = np.asarray(caps, dtype=float)
    if caps.sum() < total - 1e-12:
        raise ValueError(
            f"caps sum to {caps.sum():.4f} but {total:.4f} must be allocated - infeasible. "
            "Either the limits are too tight or the book is too big for them."
        )
    lo, hi = float(v.min() - total), float(v.max())
    for _ in range(80):          # 80 halvings exhausts double precision on any real range
        tau = 0.5 * (lo + hi)
        if np.clip(v - tau, 0.0, caps).sum() > total:
            lo = tau
        else:
            hi = tau
    return np.clip(v - 0.5 * (lo + hi), 0.0, caps)


@dataclass(frozen=True)
class Equilibrium:
    """Where the capital field came to rest."""

    weights: np.ndarray
    free_energy: float
    internal_energy: float
    entropy: float
    temperature: float
    iterations: int
    converged: bool
    binding: tuple[str, ...]           # which caps are pinned at equilibrium

    def __str__(self) -> str:
        state = "converged" if self.converged else "DID NOT CONVERGE - do not act on this"
        return (f"T={self.temperature:.3f} | F={self.free_energy:+.4f} "
                f"(U={self.internal_energy:+.4f}, S={self.entropy:.4f}) | "
                f"{self.iterations} steps, {state}"
                + (f" | caps binding: {', '.join(self.binding)}" if self.binding else ""))


class CapitalField:
    """
    Three fields over one index set, relaxed until they balance.

      hypotheses   the edge each idea claims, in bps      (what pulls capital in)
      covariance   how those ideas move together          (what pushes it apart)
      caps         the risk arbiter's hard limits          (what it may never cross)

    The free energy is F = U - T*S with

      U(w) = -edge'w + (gamma/2) w'Sigma w      the energy of an allocation
      S(w) = -sum w log w                        its entropy, i.e. diversification

    Minimising F is not a metaphor with a physics word attached. It is the same
    variational problem as the active inference loop one level up, and it has
    the same closed-form answer in the simple case: with no covariance term and
    no binding cap, the equilibrium is exactly the Boltzmann distribution
    w ∝ exp(edge / T). The solver is checked against that in the test suite.

    Temperature is the one dial. High T buys entropy - the book spreads across
    everything. Low T buys energy - the book concentrates on its best idea.
    Neither is right; the dial is the risk appetite, stated as a number instead
    of as a mood.

    Capital is CONSERVED (weights sum to the budget) and caps are HARD (enforced
    by projection). Everything else is negotiable.
    """

    def __init__(
        self,
        names: Sequence[str],
        hypotheses: Sequence[float],
        caps: Sequence[float] | float = 1.0,
        covariance: np.ndarray | None = None,
        risk_aversion: float = 1.0,
        budget: float = 1.0,
    ) -> None:
        self.names = list(names)
        self.edges = np.asarray(hypotheses, dtype=float)
        n = len(self.names)
        if self.edges.shape != (n,):
            raise ValueError(f"{n} names but {self.edges.shape} hypotheses")
        self.caps = np.full(n, float(caps)) if np.isscalar(caps) else np.asarray(caps, dtype=float)
        if self.caps.shape != (n,):
            raise ValueError(f"{n} names but {self.caps.shape} caps")
        self.covariance = np.eye(n) if covariance is None else np.asarray(covariance, dtype=float)
        self.risk_aversion = float(risk_aversion)
        self.budget = float(budget)

    # ── the thermodynamic quantities ─────────────────────────────────────────
    def internal_energy(self, w: np.ndarray) -> float:
        """U = -(what we expect to gain) + (what the covariance costs us)."""
        return float(-self.edges @ w + 0.5 * self.risk_aversion * w @ self.covariance @ w)

    def entropy(self, w: np.ndarray) -> float:
        """Shannon entropy of the allocation. Concentration is low entropy."""
        p = np.clip(w / max(w.sum(), EPS), EPS, None)
        return float(-(p * np.log(p)).sum())

    def free_energy(self, w: np.ndarray, temperature: float) -> float:
        return self.internal_energy(w) - temperature * self.entropy(w)

    def gradient(self, w: np.ndarray, temperature: float) -> np.ndarray:
        """dF/dw. The entropy term is what keeps the book from collapsing to one name."""
        safe = np.clip(w / max(w.sum(), EPS), EPS, None)
        return -self.edges + self.risk_aversion * (self.covariance @ w) \
            + temperature * (np.log(safe) + 1.0) / max(w.sum(), EPS)

    def boltzmann(self, temperature: float) -> np.ndarray:
        """
        The analytic equilibrium when covariance and caps do not bind:
        w ∝ exp(edge / T). Kept as a reference the solver is checked against.
        """
        z = self.edges / max(temperature, EPS)
        weights = np.exp(z - z.max())
        return self.budget * weights / weights.sum()

    # ── relaxation ───────────────────────────────────────────────────────────
    def equilibrium(
        self,
        temperature: float = 1.0,
        max_iterations: int = 5_000,
        step: float = 0.05,
        tolerance: float = 1e-10,
        start: np.ndarray | None = None,
    ) -> Equilibrium:
        """
        Projected gradient descent to the minimum of F.

        Deterministic - no sampling, no restarts, no seed. F is convex here
        (a convex quadratic plus a convex entropy term over a convex set), so
        the minimum is unique and the starting point does not change where it
        lands; the test suite checks that from several starts.
        """
        n = len(self.names)
        w = project_to_capped_simplex(
            np.full(n, self.budget / n) if start is None else np.asarray(start, dtype=float),
            self.caps, self.budget)
        previous = self.free_energy(w, temperature)
        converged, iterations = False, 0
        step_ceiling = max(step * 20.0, 1.0)

        for iterations in range(1, max_iterations + 1):
            candidate = project_to_capped_simplex(w - step * self.gradient(w, temperature),
                                                  self.caps, self.budget)
            current = self.free_energy(candidate, temperature)
            if current > previous:                       # overshot: halve and retry
                step *= 0.5
                if step < 1e-14:
                    converged = True                     # cannot move without going uphill
                    break
                continue
            moved = float(np.abs(candidate - w).max())
            improvement = previous - current
            w, previous = candidate, current
            # Converged when the WEIGHTS settle or when F stops improving. The
            # second criterion is the one that matters and the one that works at
            # low temperature: the entropy gradient carries a log term that
            # diverges as a weight approaches zero, so the iterates keep jittering
            # in the last few decimal places long after F has bottomed out.
            # Judging convergence by movement alone would report failure on a
            # perfectly good answer.
            if moved < tolerance or improvement < tolerance:
                converged = True
                break
            # Backtracking that only ever shrinks turns into a crawl: one early
            # overshoot and every later step is paying for it. Grow the step back
            # after a success so the solver can accelerate on easy stretches.
            step = min(step * 1.15, step_ceiling)

        binding = tuple(name for name, weight, cap in zip(self.names, w, self.caps)
                        if weight >= cap - 1e-9)
        return Equilibrium(w, self.free_energy(w, temperature), self.internal_energy(w),
                           self.entropy(w), float(temperature), iterations, converged, binding)

    def cooling_curve(self, temperatures: Sequence[float]) -> list[Equilibrium]:
        """
        Equilibria across a temperature ladder - an annealing schedule.

        Reading it top to bottom shows the book concentrating as risk appetite
        rises. It is the most honest single picture of what a strategy actually
        does with conviction.
        """
        return [self.equilibrium(temperature=t) for t in temperatures]

    def report(self, equilibrium: Equilibrium) -> str:
        lines = [str(equilibrium)]
        order = np.argsort(equilibrium.weights)[::-1]
        for i in order:
            bar = "#" * int(round(equilibrium.weights[i] * 40))
            pinned = "  <- at cap" if self.names[i] in equilibrium.binding else ""
            lines.append(f"  {self.names[i]:<14} {equilibrium.weights[i]:6.2%} "
                         f"(edge {self.edges[i]:+6.1f}) {bar}{pinned}")
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3b: TENSOR NETWORK CONTRACTION OVER FUTURE MARKET STATES
# ═════════════════════════════════════════════════════════════════════════════
#
# WHAT "QUANTUM-INSPIRED" HONESTLY MEANS HERE
#
# The machinery below - matrix product states, bond dimension, transfer
# matrices, correlation length - was invented by physicists to represent quantum
# many-body systems, and that is the whole of the borrowing. There is no quantum
# hardware, no superposition, no entanglement, and no quantum speedup. The
# numbers are ordinary non-negative probabilities and every operation is
# classical linear algebra.
#
# What IS real is the reason physicists built it. A joint distribution over N
# variables has d^N configurations: 40 binary variables is a trillion. You
# cannot enumerate that, but if the dependency structure is a chain you never
# have to - contraction answers marginals, conditionals, the single most likely
# joint state and exact samples in O(N d^2), touching each variable a constant
# number of times. That is a genuine and large win, and it is a STRUCTURAL win,
# not a hardware one.
#
# The limit, stated plainly: this is exact for chains and trees. A dependency
# graph with loops needs approximation, and this file does not implement one.
# ═════════════════════════════════════════════════════════════════════════════

NEG_INF = -np.inf


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Sum in log space without overflowing. Numpy-only, so no scipy import here."""
    values = np.asarray(values, dtype=float)
    if axis is None:
        peak = float(np.max(values)) if values.size else 0.0
        if not np.isfinite(peak):          # every entry -inf: the event is impossible
            peak = 0.0
        return float(np.log(np.sum(np.exp(values - peak))) + peak)
    peak = np.max(values, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return np.squeeze(np.log(np.sum(np.exp(values - peak), axis=axis, keepdims=True)) + peak,
                      axis=axis)


@dataclass(frozen=True)
class StateVariable:
    """One market variable and the discrete states it can take."""

    name: str
    states: tuple[str, ...]
    log_prior: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.states)

    @classmethod
    def from_probabilities(cls, name: str, probabilities: Mapping[str, float]) -> "StateVariable":
        states = tuple(probabilities)
        p = np.array([probabilities[s] for s in states], dtype=float)
        if np.any(p < 0):
            raise ValueError(f"'{name}' has a negative probability")
        if p.sum() <= 0:
            raise ValueError(f"'{name}' has no probability mass at all")
        return cls(name, states, np.log(np.clip(p / p.sum(), EPS, None)))


class MarketStateNetwork:
    """
    A chain tensor network over future market states - the probability cloud.

    Variables sit in a line; each carries a local prior, and each adjacent pair
    carries a coupling matrix saying which combinations reinforce and which
    conflict ("if liquidity is thin, a stressed regime is more likely than the
    marginals alone suggest"). The joint weight of a configuration is the
    product of all local priors and all couplings; the cloud is the normalised
    version of that.

    Everything below is one contraction of the same network:

      log_partition()      normalising constant, O(N d^2)
      marginal()           what one variable does, integrating out every other
      condition()          the cloud after news arrives
      most_likely_states() the top-k joint configurations, exactly
      sample()             exact draws, deterministic given a seed
      correlation_length() how far a shock propagates along the chain

    Contraction is exact. It is not a sampling approximation and it is not a
    mean-field one - the answers are the same numbers brute-force enumeration
    would give, which the test suite verifies directly.
    """

    def __init__(self, variables: Sequence[StateVariable],
                 couplings: Sequence[np.ndarray] | None = None) -> None:
        if len(variables) < 1:
            raise ValueError("a network needs at least one variable")
        self.variables = list(variables)
        n = len(self.variables)
        if couplings is None:
            couplings = [np.ones((self.variables[i].dim, self.variables[i + 1].dim))
                         for i in range(n - 1)]
        if len(couplings) != n - 1:
            raise ValueError(f"{n} variables need {n - 1} couplings, got {len(couplings)}")
        self.log_couplings = []
        for i, c in enumerate(couplings):
            c = np.asarray(c, dtype=float)
            expected = (self.variables[i].dim, self.variables[i + 1].dim)
            if c.shape != expected:
                raise ValueError(f"coupling {i} must be {expected}, got {c.shape}")
            if np.any(c < 0):
                raise ValueError(f"coupling {i} has a negative entry - these are weights, not amplitudes")
            self.log_couplings.append(np.log(np.clip(c, EPS, None)))

    # ── bookkeeping ──────────────────────────────────────────────────────────
    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    @property
    def n_configurations(self) -> int:
        """How many joint states exist. The number we are NOT enumerating."""
        return int(np.prod([v.dim for v in self.variables]))

    def _index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no variable named '{name}'; have {self.names}") from None

    # ── contraction ──────────────────────────────────────────────────────────
    def _messages(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Left and right partial contractions. This is the whole algorithm."""
        n = len(self.variables)
        left: list[np.ndarray] = [self.variables[0].log_prior.copy()]
        for i in range(n - 1):
            passed = _logsumexp(left[i][:, None] + self.log_couplings[i], axis=0)
            left.append(passed + self.variables[i + 1].log_prior)

        right: list[np.ndarray] = [np.zeros(self.variables[-1].dim)]
        for i in range(n - 2, -1, -1):
            incoming = self.log_couplings[i] + self.variables[i + 1].log_prior + right[0]
            right.insert(0, _logsumexp(incoming, axis=1))
        return left, right

    def log_partition(self) -> float:
        """log Z. Also log P(evidence) once the network has been conditioned."""
        left, _ = self._messages()
        return float(_logsumexp(left[-1]))

    def marginal(self, name: str) -> dict[str, float]:
        """P(this variable), with every other variable integrated out exactly."""
        i = self._index(name)
        left, right = self._messages()
        log_p = left[i] + right[i] - self.log_partition()
        return dict(zip(self.variables[i].states, np.exp(log_p)))

    def joint_marginal(self, first: str, second: str) -> np.ndarray:
        """P(x_i, x_{i+1}) for an adjacent pair - where the coupling shows itself."""
        i, j = self._index(first), self._index(second)
        if j != i + 1:
            raise ValueError(f"'{first}' and '{second}' are not adjacent; "
                             "this chain only carries couplings between neighbours")
        left, right = self._messages()
        log_p = (left[i][:, None] + self.log_couplings[i]
                 + self.variables[j].log_prior[None, :] + right[j][None, :]
                 - self.log_partition())
        return np.exp(log_p)

    def log_probability(self, configuration: Mapping[str, str]) -> float:
        """log P of one full joint configuration."""
        idx = [self.variables[i].states.index(configuration[v.name])
               for i, v in enumerate(self.variables)]
        total = sum(float(v.log_prior[k]) for v, k in zip(self.variables, idx))
        total += sum(float(self.log_couplings[i][idx[i], idx[i + 1]])
                     for i in range(len(self.variables) - 1))
        return total - self.log_partition()

    def probability(self, configuration: Mapping[str, str]) -> float:
        return float(np.exp(self.log_probability(configuration)))

    # ── news arrives ─────────────────────────────────────────────────────────
    def condition(self, **evidence: str) -> "MarketStateNetwork":
        """
        The cloud after an observation. Returns a NEW network; the original is
        untouched, so the prior stays available to compare against.
        """
        clamped = []
        for v in self.variables:
            if v.name not in evidence:
                clamped.append(v)
                continue
            observed = evidence[v.name]
            if observed not in v.states:
                raise ValueError(f"'{observed}' is not a state of '{v.name}' {v.states}")
            log_prior = np.full(v.dim, NEG_INF)
            log_prior[v.states.index(observed)] = float(v.log_prior[v.states.index(observed)])
            clamped.append(StateVariable(v.name, v.states, log_prior))
        return MarketStateNetwork(clamped, [np.exp(c) for c in self.log_couplings])

    # ── the most likely futures, exactly ─────────────────────────────────────
    def most_likely_states(self, k: int = 1) -> list[tuple[dict[str, str], float]]:
        """
        The k most probable joint configurations, found exactly by k-best
        dynamic programming over the chain - not by sampling and hoping the
        top of the cloud showed up.
        """
        n = len(self.variables)
        # beams[state] = list of (score, path) held in descending score
        beams = [[(float(self.variables[0].log_prior[a]), (a,))]
                 for a in range(self.variables[0].dim)]
        for i in range(n - 1):
            nxt = self.variables[i + 1]
            new_beams = []
            for b in range(nxt.dim):
                candidates = [
                    (score + float(self.log_couplings[i][a, b]) + float(nxt.log_prior[b]), path + (b,))
                    for a in range(self.variables[i].dim)
                    for score, path in beams[a]
                ]
                candidates.sort(key=lambda sp: sp[0], reverse=True)
                new_beams.append(candidates[:k])
            beams = new_beams

        everything = [sp for beam in beams for sp in beam]
        everything.sort(key=lambda sp: sp[0], reverse=True)
        log_z = self.log_partition()
        return [({v.name: v.states[idx] for v, idx in zip(self.variables, path)},
                 float(np.exp(score - log_z)))
                for score, path in everything[:k]]

    # ── exact draws ──────────────────────────────────────────────────────────
    def sample(self, n: int = 1, seed: int = 0) -> list[dict[str, str]]:
        """
        Exact ancestral samples from the cloud.

        Not MCMC and not rejection sampling: the right-messages already contain
        every downstream conditional, so each draw is one forward pass with the
        correct conditional at every step. Deterministic given the seed.
        """
        rng = np.random.default_rng(seed)
        left, right = self._messages()
        log_z = self.log_partition()
        out: list[dict[str, str]] = []
        for _ in range(n):
            first = np.exp(left[0] + right[0] - log_z)
            path = [int(rng.choice(len(first), p=first / first.sum()))]
            for i in range(len(self.variables) - 1):
                nxt = self.variables[i + 1]
                log_p = self.log_couplings[i][path[-1]] + nxt.log_prior + right[i + 1]
                p = np.exp(log_p - _logsumexp(log_p))
                path.append(int(rng.choice(nxt.dim, p=p / p.sum())))
            out.append({v.name: v.states[k] for v, k in zip(self.variables, path)})
        return out

    def expected_value(self, name: str, payoffs: Mapping[str, float]) -> float:
        """Expectation of a payoff defined per state of one variable."""
        marginal = self.marginal(name)
        missing = set(marginal) - set(payoffs)
        if missing:
            raise ValueError(f"no payoff given for state(s) {sorted(missing)} of '{name}'")
        return float(sum(marginal[state] * payoffs[state] for state in marginal))

    # ── physics readout ──────────────────────────────────────────────────────
    def correlation_length(self) -> float | None:
        """
        How far along the chain a shock still matters, from the transfer
        matrix's two leading eigenvalues: xi = -1 / log(|lambda_2| / |lambda_1|).

        Defined only for a homogeneous chain (same coupling everywhere), so it
        returns None otherwise rather than averaging things that should not be
        averaged. xi < 1 means neighbours barely talk; a large xi means one
        variable's state is felt across the whole network - which is what a
        contagion looks like before anyone calls it one.
        """
        if len(self.variables) < 2:
            return None
        first = self.log_couplings[0]
        if not all(c.shape == first.shape and np.allclose(c, first) for c in self.log_couplings):
            return None
        if not all(v.dim == self.variables[0].dim
                   and np.allclose(v.log_prior, self.variables[0].log_prior)
                   for v in self.variables):
            return None
        transfer = np.exp(first) * np.exp(self.variables[0].log_prior)[None, :]
        eigenvalues = np.sort(np.abs(np.linalg.eigvals(transfer)))[::-1]
        if len(eigenvalues) < 2 or eigenvalues[0] <= 0 or eigenvalues[1] <= 0:
            return None
        ratio = eigenvalues[1] / eigenvalues[0]
        return float(-1.0 / np.log(ratio)) if 0 < ratio < 1 else float("inf")

    # ── brute force, for verification only ───────────────────────────────────
    def enumerate_exactly(self) -> tuple[list[dict[str, str]], np.ndarray]:
        """
        Every configuration and its probability, by explicit enumeration.

        Exponential and therefore useless in production - it exists so the test
        suite can prove the contraction returns the same numbers. Refuses to run
        past 2^16 configurations, because a verification tool that hangs is not
        a verification tool.
        """
        if self.n_configurations > 65_536:
            raise ValueError(
                f"{self.n_configurations:,} configurations is too many to enumerate - "
                "which is the entire reason contraction exists"
            )
        import itertools as _it
        configs, log_weights = [], []
        for combo in _it.product(*[range(v.dim) for v in self.variables]):
            weight = sum(float(v.log_prior[k]) for v, k in zip(self.variables, combo))
            weight += sum(float(self.log_couplings[i][combo[i], combo[i + 1]])
                          for i in range(len(self.variables) - 1))
            configs.append({v.name: v.states[k] for v, k in zip(self.variables, combo)})
            log_weights.append(weight)
        log_weights = np.array(log_weights)
        return configs, np.exp(log_weights - _logsumexp(log_weights))

    def report(self) -> str:
        lines = [f"{len(self.variables)} variables, {self.n_configurations:,} joint configurations, "
                 f"contracted in O(N d^2) without enumerating any of them"]
        for v in self.variables:
            marginal = self.marginal(v.name)
            body = "  ".join(f"{state} {p:5.1%}" for state, p in marginal.items())
            lines.append(f"  {v.name:<12} {body}")
        xi = self.correlation_length()
        if xi is not None:
            lines.append(f"  correlation length {xi:.2f} variables")
        return "\n".join(lines)


def synth_panel(n: int = 300, drift: float = 0.0, vol: float = 0.01,
                shock: float = 0.0, seed: int = 0) -> dict[str, np.ndarray]:
    """Deterministic synthetic asset for demos and tests. Not market data."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    if shock:
        rets[-8:] += shock  # a recent dislocation
    price = 100.0 * np.exp(np.cumsum(rets))
    return {"price": price, "return": rets, "volume": np.abs(rets) * 1e6 + 1e5}


if __name__ == "__main__":
    print("STARK MK18 - market tensors demo\n")
    tensor = MarketTensor.from_panels({
        "CALM":     synth_panel(vol=0.004, seed=1),
        "HOT":      synth_panel(vol=0.030, seed=2),
        "STRETCH":  synth_panel(vol=0.006, shock=0.012, seed=3),
        "CRASHING": synth_panel(vol=0.020, shock=-0.030, seed=4),
    })
    print(tensor.report())
    print("\nNote CALM vs HOT: a similar stretch in a hot tape earns a smaller F.")
    print("That is the field doing its job - it refuses to be impressed by noise.")

    # ── Phase 3a: capital at equilibrium ─────────────────────────────────────
    print("\n" + "=" * 78)
    print(" PHASE 3a - CAPITAL FIELD AT THERMODYNAMIC EQUILIBRIUM")
    print("=" * 78)

    field = CapitalField(
        names=["momentum", "carry", "mean_rev", "vol_arb", "crowded_beta"],
        hypotheses=[18.0, 12.0, 9.0, 6.0, 22.0],      # est. edge in bps, from the generator
        caps=[0.25, 0.25, 0.25, 0.25, 0.10],          # hard limits, from the risk arbiter
        risk_aversion=1.0,
    )
    print("\nThree fields over one index set: what each idea claims (edge), what the")
    print("book may hold (caps), and where capital settles between them.\n")
    for temperature in (20.0, 8.0, 3.0, 1.0):
        equilibrium = field.equilibrium(temperature=temperature)
        print(field.report(equilibrium))
        print()
    print("Cooling concentrates the book on its best ideas - until the caps stop it.")
    print("The 10% limit on crowded_beta binds at EVERY temperature: the strongest")
    print("hypothesis in the set never gets more than the arbiter allows, no matter how")
    print("cold it gets or how good it looks. That is rule 2, expressed as a projection.")

    # ── Phase 3b: the probability cloud ──────────────────────────────────────
    print("\n" + "=" * 78)
    print(" PHASE 3b - TENSOR NETWORK CONTRACTION OVER FUTURE STATES")
    print("=" * 78)

    cloud = MarketStateNetwork(
        [StateVariable.from_probabilities("liquidity", {"thin": 0.30, "normal": 0.50, "deep": 0.20}),
         StateVariable.from_probabilities("regime", {"calm": 0.60, "stressed": 0.40}),
         StateVariable.from_probabilities("direction", {"down": 0.35, "flat": 0.30, "up": 0.35}),
         StateVariable.from_probabilities("flow", {"selling": 0.4, "balanced": 0.3, "buying": 0.3})],
        couplings=[
            np.array([[0.5, 2.0], [1.0, 1.0], [2.0, 0.5]]),        # thin liquidity favours stress
            np.array([[1.0, 1.2, 1.0], [1.6, 0.8, 0.6]]),          # stress skews to the downside
            np.array([[2.0, 1.0, 0.5], [1.0, 1.5, 1.0], [0.5, 1.0, 2.0]]),   # direction follows flow
        ],
    )
    print("\n" + cloud.report())

    print("\nNews arrives: liquidity is observed THIN. The whole cloud updates at once.")
    conditioned = cloud.condition(liquidity="thin")
    for name in ("regime", "direction", "flow"):
        before, after = cloud.marginal(name), conditioned.marginal(name)
        shifts = "  ".join(f"{s} {before[s]:5.1%}->{after[s]:5.1%}" for s in before)
        print(f"  {name:<11} {shifts}")

    print("\nThe most likely joint futures (exact, not sampled):")
    for configuration, probability in cloud.most_likely_states(4):
        body = ", ".join(f"{k}={v}" for k, v in configuration.items())
        print(f"  {probability:6.2%}  {body}")

    print("\nAgreement with brute force on this small network:")
    configurations, probabilities = cloud.enumerate_exactly()
    worst = max(abs(cloud.probability(c) - p) for c, p in zip(configurations, probabilities))
    print(f"  {len(configurations)} configurations enumerated, worst disagreement {worst:.2e}")
    print("  Contraction is EXACT. It is not a sampling approximation.")

    # ── and the reason any of this matters ───────────────────────────────────
    import time
    big = MarketStateNetwork(
        [StateVariable.from_probabilities(f"v{i}", {"lo": 0.45, "hi": 0.55}) for i in range(40)],
        couplings=[np.array([[1.6, 0.6], [0.6, 1.6]])] * 39,       # each state favours its neighbour
    )
    started = time.perf_counter()
    marginal = big.marginal("v20")
    top = big.most_likely_states(1)[0]
    elapsed = time.perf_counter() - started
    print(f"\n40 variables: {big.n_configurations:,} joint configurations.")
    print(f"  middle variable marginal {dict((k, round(v, 4)) for k, v in marginal.items())}")
    print(f"  most likely joint state has probability {top[1]:.2e}")
    print(f"  correlation length {big.correlation_length():.2f} variables")
    print(f"  time taken: {elapsed * 1000:.1f} ms, without enumerating a single configuration.")
    print("\nThat is the whole argument for tensor networks, and it is structural, not")
    print("quantum: no superposition, no quantum hardware, no quantum speedup. Just a")
    print("chain contracted in O(N d^2) instead of a trillion states visited one by one.")
