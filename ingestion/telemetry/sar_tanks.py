#!/usr/bin/env python3
"""
Phase 4 - SAR floating-roof tank gauging.

THE METHOD, WHICH IS REAL

A crude oil storage tank has a roof that floats on the oil, so the roof sits at
the level of the liquid. Look at it obliquely and the tank's rim casts a shadow
onto that roof. The width of that shadow, plus the geometry of the look angle,
gives the roof's depth below the rim - and depth below the rim is how empty the
tank is. Radar sees through cloud and at night, so unlike optical imagery it
works every pass. This is not speculative; it is how commercial oil-inventory
services operate.

WHAT THIS FILE DOES AND DOES NOT DO

  Does:     the geometry, the slant-to-ground-range conversion, propagation of
            measurement error into a barrel figure, and aggregation across many
            tanks with the error scaling that implies.
  Does not: touch a satellite. There is no imagery pipeline here, no
            segmentation, no shadow finder. Shadow width arrives as a measured
            input, from you.

THE NUMBER THAT DECIDES WHETHER THIS IS USEFUL

Depth below rim d = w * cot(theta), so the depth error is the width error times
cot(theta). At Sentinel-1's free 10 m resolution and a 35 degree incidence
angle, ONE pixel of shadow-edge error is 14 metres of depth - most of a tank's
height. A single-tank fill estimate from free imagery is close to worthless, and
this module will tell you so rather than printing a confident percentage.

What rescues it is arithmetic: independent errors across N tanks shrink like
sqrt(N), so a basin of 400 tanks can be gauged to a few percent from imagery
that cannot gauge any one of them at all. Sub-metre commercial SAR fixes the
single-tank case directly, at a price.

numpy only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

CUBIC_METRES_TO_BARRELS = 6.2898


@dataclass(frozen=True)
class Tank:
    """A floating-roof tank. Radius and wall height come from a survey, not the image."""

    tank_id: str
    radius_m: float
    wall_height_m: float
    latitude: float = 0.0
    longitude: float = 0.0
    radius_sigma_m: float = 0.5
    height_sigma_m: float = 0.5

    @property
    def capacity_m3(self) -> float:
        return float(math.pi * self.radius_m ** 2 * self.wall_height_m)

    @property
    def capacity_barrels(self) -> float:
        return self.capacity_m3 * CUBIC_METRES_TO_BARRELS


@dataclass(frozen=True)
class SarLook:
    """The imaging geometry of one SAR observation."""

    incidence_deg: float          # from vertical; Sentinel-1 IW is roughly 30-46
    pixel_spacing_m: float        # 10.0 for Sentinel-1 IW GRD, ~1.0 for commercial
    slant_range: bool = False     # True if the shadow was measured in slant range
    incidence_sigma_deg: float = 0.5
    timestamp: float = 0.0

    @property
    def incidence_rad(self) -> float:
        return math.radians(self.incidence_deg)

    def ground_width_m(self, shadow_pixels: float) -> float:
        """Shadow width in metres on the ground. Slant range needs dividing by sin(theta)."""
        width = shadow_pixels * self.pixel_spacing_m
        return width / math.sin(self.incidence_rad) if self.slant_range else width

    @property
    def depth_per_metre_of_shadow(self) -> float:
        """cot(theta): how many metres of depth one metre of shadow error buys you."""
        return 1.0 / math.tan(self.incidence_rad)


@dataclass(frozen=True)
class FillEstimate:
    """A fill reading with the error bar that decides whether it means anything."""

    tank_id: str
    fill_fraction: float
    fill_sigma: float
    roof_depth_m: float
    volume_barrels: float
    volume_sigma_barrels: float
    look: SarLook
    saturated: bool = False        # geometry pushed the answer outside [0, 1]

    @property
    def usable(self) -> bool:
        """
        Is one reading worth anything on its own?

        A fill estimate whose 1-sigma error exceeds 10% of the tank cannot
        distinguish half-full from nearly-full, and should only ever be used
        inside an aggregate.
        """
        return self.fill_sigma <= 0.10 and not self.saturated

    def __str__(self) -> str:
        verdict = "usable alone" if self.usable else "AGGREGATE ONLY - too noisy to stand alone"
        note = " [SATURATED]" if self.saturated else ""
        return (f"{self.tank_id}: {self.fill_fraction:5.1%} +/- {self.fill_sigma:.1%} "
                f"({self.volume_barrels:,.0f} +/- {self.volume_sigma_barrels:,.0f} bbl) "
                f"{verdict}{note}")


def estimate_fill(tank: Tank, look: SarLook, shadow_pixels: float,
                  shadow_sigma_pixels: float = 1.0) -> FillEstimate:
    """
    Fill fraction from shadow width, with first-order error propagation.

        d    = w * cot(theta)              roof depth below the rim
        fill = (H - d) / H

    Errors travel through the partial derivatives:

        dd/dw     = cot(theta)
        dd/dtheta = -w * csc^2(theta)

    and the tank's own survey uncertainty in H is folded in. Half a pixel of
    edge ambiguity is optimistic for real SAR; one pixel is the honest default.
    """
    if tank.wall_height_m <= 0:
        raise ValueError(f"{tank.tank_id}: wall height must be positive")
    if not 0 < look.incidence_deg < 90:
        raise ValueError(f"incidence angle {look.incidence_deg} is outside (0, 90)")

    theta = look.incidence_rad
    width = look.ground_width_m(shadow_pixels)
    width_sigma = look.ground_width_m(shadow_sigma_pixels)

    depth = width / math.tan(theta)
    d_dw = 1.0 / math.tan(theta)
    d_dtheta = -width / (math.sin(theta) ** 2)
    depth_variance = (d_dw * width_sigma) ** 2 + \
                     (d_dtheta * math.radians(look.incidence_sigma_deg)) ** 2
    depth_sigma = math.sqrt(max(depth_variance, 0.0))

    raw_fill = 1.0 - depth / tank.wall_height_m
    fill = float(np.clip(raw_fill, 0.0, 1.0))
    saturated = not (0.0 <= raw_fill <= 1.0)

    # fill = 1 - d/H, so both d and H carry error into it
    fill_variance = (depth_sigma / tank.wall_height_m) ** 2 + \
                    (depth * tank.height_sigma_m / tank.wall_height_m ** 2) ** 2
    fill_sigma = math.sqrt(max(fill_variance, 0.0))

    volume_m3 = math.pi * tank.radius_m ** 2 * tank.wall_height_m * fill
    # relative error adds the radius term twice: volume goes as r^2
    relative = math.sqrt((fill_sigma / max(fill, 1e-9)) ** 2 +
                         (2 * tank.radius_sigma_m / tank.radius_m) ** 2) if fill > 0 else 0.0
    volume_barrels = volume_m3 * CUBIC_METRES_TO_BARRELS

    return FillEstimate(tank.tank_id, fill, fill_sigma, depth, volume_barrels,
                        volume_barrels * relative, look, saturated)


def shadow_pixels_for_fill(tank: Tank, look: SarLook, fill_fraction: float) -> float:
    """
    The forward model: what shadow would a tank at this fill actually cast?

    Inverse of estimate_fill, used to test the round trip and to work out whether
    a given sensor can resolve the difference you care about BEFORE you buy the
    imagery.
    """
    depth = tank.wall_height_m * (1.0 - fill_fraction)
    width = depth * math.tan(look.incidence_rad)
    if look.slant_range:
        width *= math.sin(look.incidence_rad)
    return width / look.pixel_spacing_m


def resolvable_fill_step(tank: Tank, look: SarLook, shadow_sigma_pixels: float = 1.0) -> float:
    """
    The smallest fill change this sensor can see on this tank, at 1 sigma.

    Ask this first. If the answer is 0.7, the sensor cannot tell a half-empty
    tank from a full one and no amount of processing will change that.
    """
    depth_sigma = look.ground_width_m(shadow_sigma_pixels) / math.tan(look.incidence_rad)
    return float(depth_sigma / tank.wall_height_m)


@dataclass(frozen=True)
class BasinInventory:
    """Aggregate inventory across many tanks, where the errors finally behave."""

    tanks: int
    barrels: float
    barrels_sigma: float
    usable_alone: int
    capacity_barrels: float
    saturated: int = 0

    @property
    def utilisation(self) -> float:
        return self.barrels / self.capacity_barrels if self.capacity_barrels else 0.0

    @property
    def relative_error(self) -> float:
        return self.barrels_sigma / self.barrels if self.barrels else float("inf")

    @property
    def biased(self) -> bool:
        """
        Did enough tanks hit a physical bound to bias the total?

        Clipping is not symmetric. When the shadow noise is large relative to
        the shadow itself, readings pile up against fill=0 and fill=1 and the
        sum drifts - in a direction the error bar knows nothing about.
        """
        return self.tanks > 0 and self.saturated / self.tanks > 0.05

    def warnings(self) -> list[str]:
        if not self.biased:
            return []
        return [
            f"{self.saturated}/{self.tanks} tanks ({self.saturated / self.tanks:.0%}) hit a "
            "physical bound. Their readings were clipped, which BIASES this total. "
            "Averaging kills variance, not bias - the error bar below does not cover it. "
            "Use a finer sensor or a shallower look angle."
        ]

    def __str__(self) -> str:
        head = (f"{self.tanks} tanks | {self.barrels / 1e6:,.2f} +/- "
                f"{self.barrels_sigma / 1e6:,.2f} million bbl "
                f"({self.relative_error:.1%}) | {self.utilisation:.1%} of capacity | "
                f"{self.usable_alone} tanks readable on their own")
        return head + ("".join(f"\n  ! {w}" for w in self.warnings()))


def aggregate_inventory(estimates: Sequence[FillEstimate],
                        tanks: Sequence[Tank]) -> BasinInventory:
    """
    Sum a basin. Errors add in quadrature, which is the entire economics of this.

    Independent per-tank errors mean the aggregate's RELATIVE error falls like
    1/sqrt(N): 400 tanks that are individually hopeless become an aggregate good
    to a few percent - PROVIDED the errors are zero-mean and independent.

    Two ways that fails, neither visible in the error bar:

      clipping     a fill cannot be below 0 or above 1, so when the noise is
                   large relative to the shadow, readings pile up against those
                   bounds and the sum drifts. Counted here as `saturated`, and
                   warned about, because averaging kills variance, not bias.
      systematics  a mis-calibrated incidence angle or a processing bias applied
                   to the whole scene shifts every tank the same way and does not
                   average away at all. This function cannot see it. Vary the
                   geometry across passes and compare.
    """
    if len(estimates) != len(tanks):
        raise ValueError("one estimate per tank, please")
    barrels = float(sum(e.volume_barrels for e in estimates))
    sigma = float(math.sqrt(sum(e.volume_sigma_barrels ** 2 for e in estimates)))
    return BasinInventory(len(estimates), barrels, sigma,
                          sum(e.usable for e in estimates),
                          float(sum(t.capacity_barrels for t in tanks)),
                          sum(e.saturated for e in estimates))


if __name__ == "__main__":
    print("STARK MK18 - SAR floating-roof tank gauging\n")

    tank = Tank("CUSHING-042", radius_m=40.0, wall_height_m=20.0)
    print(f"tank: {tank.radius_m:.0f} m radius, {tank.wall_height_m:.0f} m tall, "
          f"capacity {tank.capacity_barrels:,.0f} bbl\n")

    sentinel = SarLook(incidence_deg=35.0, pixel_spacing_m=10.0)
    commercial = SarLook(incidence_deg=35.0, pixel_spacing_m=1.0, incidence_sigma_deg=0.2)

    print("Before buying any imagery, ask what it can resolve:")
    for label, look in (("Sentinel-1 (free, 10 m)", sentinel), ("commercial (1 m)", commercial)):
        step = resolvable_fill_step(tank, look)
        print(f"  {label:<24} smallest resolvable fill change: {step:6.1%}"
              + ("   <- cannot tell half-full from full" if step > 0.25 else ""))

    print("\nSame tank, genuinely 70% full, measured by each sensor:")
    for label, look in (("Sentinel-1", sentinel), ("commercial", commercial)):
        pixels = shadow_pixels_for_fill(tank, look, 0.70)
        estimate = estimate_fill(tank, look, pixels)
        print(f"  {label:<12} shadow {pixels:5.2f} px -> {estimate}")
    print("  Both recover 70%. Only one of them KNOWS 70%.")

    print("\nNow a whole basin of 400 tanks, measured by each sensor:")
    rng = np.random.default_rng(4)
    basin = [Tank(f"T-{i:03d}", radius_m=40.0, wall_height_m=20.0) for i in range(400)]
    true_fills = rng.uniform(0.35, 0.85, len(basin))
    truth = sum(t.capacity_barrels * f for t, f in zip(basin, true_fills))

    for label, look in (("Sentinel-1 (10 m)", sentinel), ("commercial (1 m)", commercial)):
        noise = np.random.default_rng(4).normal(0, 1.0, len(basin))
        estimates = [
            estimate_fill(one, look, max(shadow_pixels_for_fill(one, look, fill) + n, 0.0))
            for one, fill, n in zip(basin, true_fills, noise)
        ]
        inventory = aggregate_inventory(estimates, basin)
        error = (inventory.barrels - truth) / truth
        print(f"\n  {label}")
        print(f"  {inventory}")
        print(f"  true total {truth / 1e6:,.2f} million bbl -> off by {error:+.2%}")

    print("\n  Read those two carefully. The 1 m sensor lands inside its error bar. The")
    print("  free one does NOT - it is low by more than its stated uncertainty, and no")
    print("  amount of extra tanks fixes that, because the miss is bias, not variance.")
    print("  Its shadow is a fraction of a pixel wide, so noise pushes reading after")
    print("  reading against the physical bounds and the clipped total drifts.")
    print("\n  sqrt(N) averaging is a real and powerful effect. It buys down variance and")
    print("  nothing else. A pipeline that quotes the aggregate error bar without checking")
    print("  the saturation count is quoting a number it has not earned.")
