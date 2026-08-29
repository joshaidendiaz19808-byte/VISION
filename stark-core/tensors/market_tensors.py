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
