#!/usr/bin/env python3
"""
Level 1 - Neural intuition & hypotheses.

The generator is the part of STARK that is *allowed* to guess. It looks at a
block of features and proposes falsifiable claims about them, ranked by how
promising they look.

Honest mechanic behind the "intuition":
  rank correlation between a feature and forward returns, shrunk toward zero by
  sample size, then blended with a pluggable model score. Nothing here is
  evidence - it is a to-do list for the red team.

Every number this module emits is a PRIOR. Priors are cheap. They are not
permission to trade anything.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

# Confidence is shrunk toward zero until we have roughly this many observations.
SHRINKAGE_N = 50

# A scorer maps (feature_name, feature_values, forward_returns) -> [0, 1].
Scorer = Callable[[str, np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class Hypothesis:
    """One falsifiable claim about one feature."""

    id: str
    claim: str
    feature: str
    direction: int          # +1 = high feature -> high return, -1 = inverted
    horizon: int            # bars ahead the claim is about
    prior: float            # shrunk confidence in [0, 1] - NOT a probability of profit
    raw_ic: float           # unshrunk rank correlation
    n_obs: int
    n_considered: int       # how many features were searched to find this one
    edge_bps: float = 0.0   # est. GROSS edge - costs are the red team's business
    vol_bps: float = 0.0    # per-period sigma, the arbiter's variance input
    tags: tuple[str, ...] = ()

    def __str__(self) -> str:
        return (
            f"[{self.id}] {self.claim}\n"
            f"        prior {self.prior:.2f} | rank IC {self.raw_ic:+.3f} | "
            f"est. {self.edge_bps:.1f} bps at {self.vol_bps:.0f} bps vol | "
            f"n={self.n_obs} | searched {self.n_considered} features"
        )


def rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. Robust to the fat tails that break Pearson."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return 0.0
    xr = _ranks(x[ok])
    yr = _ranks(y[ok])
    xs, ys = xr.std(), yr.std()
    if xs == 0 or ys == 0:
        return 0.0
    return float(np.mean((xr - xr.mean()) * (yr - yr.mean())) / (xs * ys))


def _ranks(v: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared."""
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=float)
    ranks[order] = np.arange(len(v), dtype=float)
    # average tied ranks so a constant column cannot fake a correlation
    uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return ranks


def shrink(ic: float, n: int, k: int = SHRINKAGE_N) -> float:
    """
    Pull |IC| toward zero when the sample is thin.

    n=10  -> keeps ~17% of the signal
    n=250 -> keeps ~83%
    This is the honesty rule in one line: thin data cannot buy confidence.
    """
    if n <= 0:
        return 0.0
    return float(abs(ic) * n / (n + k))


def estimate_edge_bps(ic: float, forward_returns: np.ndarray) -> float:
    """
    Crude but honest gross-edge estimate: |IC| x the spread of returns, in bps.

    Rank correlation says how reliably a feature orders returns, not how much
    money that ordering is worth; multiplying by return dispersion converts one
    into the other. It is an estimate and every caller labels it `est.` - but it
    is defined ONCE, here, so the red team, the arbiter and JARVIS can never
    quietly disagree about what the edge is.
    """
    y = np.asarray(forward_returns, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0
    return float(abs(ic) * float(np.std(y)) * 10_000.0)


def estimate_vol_bps(forward_returns: np.ndarray) -> float:
    """Per-period sigma of the strategy's returns, in bps. The arbiter's variance input."""
    y = np.asarray(forward_returns, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 2:
        return 0.0
    return float(np.std(y) * 10_000.0)


def _heuristic_intuition(name: str, values: np.ndarray, fwd: np.ndarray) -> float:
    """
    Default stand-in for the neural model.

    Replace this with a real learned scorer; the contract is the only thing that
    matters. It rewards signal strength and penalises features that are mostly
    flat (a feature that barely moves cannot be driving anything).
    """
    ic = abs(rank_ic(values, fwd))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    spread = float(np.std(finite))
    liveliness = spread / (spread + 1e-9 + float(np.mean(np.abs(finite))) * 0.1)
    return float(np.clip(ic * (0.5 + 0.5 * liveliness), 0.0, 1.0))


class Generator:
    """
    Level 1 idea factory.

    Deterministic given a seed. Produces at most `max_hypotheses` claims, each
    tagged with how many features were searched so the red team can charge us
    for the search (see red_team.attack_multiple_testing).
    """

    def __init__(
        self,
        seed: int = 0,
        scorer: Scorer | None = None,
        max_hypotheses: int = 8,
        min_prior: float = 0.02,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.scorer: Scorer = scorer or _heuristic_intuition
        self.max_hypotheses = max_hypotheses
        self.min_prior = min_prior

    def generate(
        self,
        features: Mapping[str, Sequence[float]],
        forward_returns: Sequence[float],
        horizon: int = 1,
    ) -> list[Hypothesis]:
        """Rank every feature against forward returns and keep the best few."""
        fwd = np.asarray(forward_returns, dtype=float)
        n_considered = len(features)
        out: list[Hypothesis] = []

        for name, raw in features.items():
            values = np.asarray(raw, dtype=float)
            if len(values) != len(fwd):
                raise ValueError(
                    f"feature '{name}' has {len(values)} rows, forward_returns has {len(fwd)}"
                )
            ic = rank_ic(values, fwd)
            n = int((np.isfinite(values) & np.isfinite(fwd)).sum())
            intuition = float(self.scorer(name, values, fwd))
            prior = min(shrink(ic, n), intuition)  # the more sceptical of the two
            if prior < self.min_prior:
                continue
            direction = 1 if ic >= 0 else -1
            claim = self._claim(name, direction, horizon)
            out.append(
                Hypothesis(
                    id=_short_id(f"{name}|{direction}|{horizon}"),
                    claim=claim,
                    feature=name,
                    direction=direction,
                    horizon=horizon,
                    prior=round(prior, 4),
                    raw_ic=round(ic, 4),
                    n_obs=n,
                    n_considered=n_considered,
                    edge_bps=round(estimate_edge_bps(ic, fwd), 4),
                    vol_bps=round(estimate_vol_bps(fwd), 4),
                    tags=("unverified",),
                )
            )

        out.sort(key=lambda h: h.prior, reverse=True)
        return out[: self.max_hypotheses]

    @staticmethod
    def _claim(feature: str, direction: int, horizon: int) -> str:
        side = "rises" if direction > 0 else "falls"
        return (
            f"When '{feature}' is high, the next {horizon}-bar return {side} "
            f"more often than chance."
        )


def _short_id(text: str) -> str:
    return "H-" + hashlib.sha256(text.encode()).hexdigest()[:6].upper()


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 400
    real = rng.normal(0, 1, n)
    noise = {f"noise_{i}": rng.normal(0, 1, n) for i in range(6)}
    fwd = (0.30 * real + rng.normal(0, 1, n)) * 0.01   # ~1% per-bar returns

    gen = Generator(seed=1)
    ideas = gen.generate({"real_signal": real, **noise}, fwd)

    print("STARK MK18 - generator demo")
    print(f"searched {len(noise) + 1} features, kept {len(ideas)}\n")
    for h in ideas:
        print(" ", h, "\n")
    print("None of these are evidence yet. Hand them to red_team.py.")
