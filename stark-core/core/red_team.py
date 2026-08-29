#!/usr/bin/env python3
"""
Level 1 - Adversarial counter-agents.

The generator's job is to be interesting. The red team's job is to kill it.

Six attacks, each one a way a hypothesis is usually wrong:
  1. out-of-sample     - does the edge survive on data it never saw?
  2. stability         - is it present across folds, or one lucky window?
  3. multiple testing  - did we find it, or did we search until something showed?
  4. permutation       - can shuffled data produce this by accident?
  5. cost              - is the edge bigger than the fee to harvest it?
  6. regime            - does it exist in more than one volatility state?

A hypothesis that survives is not "true". It is *not yet dead*, which is the
strongest thing a strategy is ever allowed to be.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

if __package__ in (None, ""):  # run as a script: put the stark-core/ root on the path
    import pathlib as _pl, sys as _sys
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

from core.generator import Hypothesis, rank_ic

# An attack this severe kills the hypothesis outright.
KILL_THRESHOLD = 0.6


@dataclass(frozen=True)
class Attack:
    """One adversarial probe and what it found."""

    name: str
    survived: bool
    severity: float          # 0 = harmless, 1 = fatal
    finding: str
    fix: str | None = None

    def __str__(self) -> str:
        mark = "OK  " if self.survived else "KILL" if self.severity >= KILL_THRESHOLD else "WARN"
        return f"  [{mark}] {self.name}: {self.finding}"


@dataclass
class RedTeamReport:
    hypothesis_id: str
    attacks: list[Attack] = field(default_factory=list)

    @property
    def kill_shots(self) -> list[Attack]:
        return [a for a in self.attacks if not a.survived and a.severity >= KILL_THRESHOLD]

    @property
    def survived(self) -> bool:
        return not self.kill_shots

    @property
    def damage(self) -> float:
        """Worst severity taken. Used to discount the generator's prior."""
        return max((a.severity for a in self.attacks), default=0.0)

    def adjusted_confidence(self, prior: float) -> float:
        """Conviction may only ever go DOWN here (CLAUDE.md rule 7)."""
        if not self.survived:
            return 0.0
        return round(float(prior) * (1.0 - self.damage), 4)

    def fixes(self) -> list[str]:
        return [a.fix for a in self.attacks if a.fix and not a.survived]


# ── attacks ───────────────────────────────────────────────────────────────────

def attack_out_of_sample(x: np.ndarray, y: np.ndarray, split: float = 0.7) -> Attack:
    k = int(len(x) * split)
    if k < 20 or len(x) - k < 20:
        return Attack("out-of-sample", False, 0.7, "Not enough rows to hold anything out.",
                      "Collect more history before believing this.")
    ins, oos = rank_ic(x[:k], y[:k]), rank_ic(x[k:], y[k:])
    flipped = np.sign(ins) != np.sign(oos)
    decay = 1.0 if abs(ins) < 1e-9 else max(0.0, 1.0 - abs(oos) / abs(ins))
    if flipped:
        return Attack("out-of-sample", False, 0.9,
                      f"Edge inverts out of sample (in {ins:+.3f} -> out {oos:+.3f}).",
                      "This is the signature of a curve fit. Discard it.")
    if decay > 0.5:
        return Attack("out-of-sample", False, 0.5,
                      f"Edge decays {decay*100:.0f}% out of sample ({ins:+.3f} -> {oos:+.3f}).",
                      "Halve your size expectation, or find why the recent regime is different.")
    return Attack("out-of-sample", True, round(float(decay) * 0.4, 3),
                  f"Holds out of sample ({ins:+.3f} -> {oos:+.3f}).")


def attack_stability(x: np.ndarray, y: np.ndarray, folds: int = 5) -> Attack:
    if len(x) < folds * 20:
        return Attack("stability", False, 0.6, f"Too few rows for {folds} folds.",
                      "Need ~100+ rows minimum before folds mean anything.")
    ics = [rank_ic(xa, ya) for xa, ya in zip(np.array_split(x, folds), np.array_split(y, folds))]
    dominant = np.sign(np.mean(ics))
    agree = float(np.mean([np.sign(i) == dominant for i in ics]))
    if agree < 0.6:
        return Attack("stability", False, 0.7,
                      f"Sign flips across folds - only {agree*100:.0f}% agree ({', '.join(f'{i:+.2f}' for i in ics)}).",
                      "One window is carrying the whole result. Treat it as noise.")
    return Attack("stability", True, round(1.0 - agree, 3),
                  f"{agree*100:.0f}% of folds agree on direction.")


def attack_multiple_testing(ic: float, n: int, n_considered: int) -> Attack:
    """
    Charge the hypothesis for the size of the search that found it.

    Sidak: a 5% test repeated over m features fires by luck ~1-(0.95^m) of the
    time. Searching 40 features means the bar has to be much higher than one.
    """
    if n < 3:
        return Attack("multiple-testing", False, 0.8, "Sample too small to test at all.", None)
    t = abs(ic) * np.sqrt(max(n - 2, 1) / max(1e-9, 1 - min(ic**2, 0.999999)))
    p_raw = 2.0 * (1.0 - _norm_cdf(t))
    p_adj = 1.0 - (1.0 - min(p_raw, 1.0)) ** max(n_considered, 1)
    if p_adj > 0.05:
        return Attack("multiple-testing", False, 0.75,
                      f"Dies once you pay for the search: p={p_raw:.3f} raw, "
                      f"p={p_adj:.3f} after {n_considered} features.",
                      "Pre-register one hypothesis and test it on fresh data instead of ranking many.")
    return Attack("multiple-testing", True, round(min(p_adj / 0.05, 1.0) * 0.3, 3),
                  f"Survives the search penalty (p={p_adj:.4f} over {n_considered} features).")


def attack_permutation(x: np.ndarray, y: np.ndarray, n_sims: int = 2000, seed: int = 42) -> Attack:
    """Shuffle the labels. If noise reproduces this often, it IS noise."""
    rng = np.random.default_rng(seed)
    observed = abs(rank_ic(x, y))
    hits = 0
    for _ in range(n_sims):
        if abs(rank_ic(x, rng.permutation(y))) >= observed:
            hits += 1
    p = (hits + 1) / (n_sims + 1)
    if p > 0.05:
        return Attack("permutation", False, 0.8,
                      f"Shuffled data reproduces this {p*100:.1f}% of the time.",
                      "Random data clears your bar. There is nothing here to trade.")
    return Attack("permutation", True, round(min(p / 0.05, 1.0) * 0.3, 3),
                  f"Shuffled data reproduces it only {p*100:.2f}% of the time.")


def attack_cost(y: np.ndarray, direction: int, ic: float, cost_bps: float = 5.0) -> Attack:
    """
    Turn a rank correlation into a crude bps expectation and pay the toll.

    Rough but honest: expected edge ~ |IC| x the spread of returns.
    Labelled est. because that is what it is.
    """
    spread_bps = float(np.std(y)) * 10_000.0
    gross = abs(ic) * spread_bps
    net = gross - cost_bps
    if net <= 0:
        return Attack("cost", False, 0.85,
                      f"Est. gross edge {gross:.1f} bps < {cost_bps:.0f} bps of cost. Wins on paper, loses live.",
                      "Trade it less often, hold longer, or use limit orders. Or drop it.")
    if net < cost_bps:
        return Attack("cost", False, 0.4,
                      f"Est. net edge only {net:.1f} bps after {cost_bps:.0f} bps cost - one bad fill erases it.",
                      "Size it small and measure real slippage before scaling.")
    return Attack("cost", True, round(float(cost_bps / max(gross, 1e-9)) * 0.4, 3),
                  f"Est. net edge {net:.1f} bps after {cost_bps:.0f} bps cost.")


def attack_regime(x: np.ndarray, y: np.ndarray, window: int = 20) -> Attack:
    """Split by realised volatility. An edge in exactly one regime is a bet on that regime."""
    if len(x) < window * 3:
        return Attack("regime", False, 0.5, "Too few rows to separate regimes.",
                      "Need at least 3 volatility windows of data.")
    vol = _rolling_std(y, window)
    lo, hi = np.nanquantile(vol, [1 / 3, 2 / 3])
    buckets = {"calm": vol <= lo, "normal": (vol > lo) & (vol <= hi), "stressed": vol > hi}
    ics = {k: rank_ic(x[m], y[m]) for k, m in buckets.items() if m.sum() >= 15}
    if not ics:
        return Attack("regime", False, 0.5, "No regime had enough rows to test.", None)
    dominant = np.sign(np.mean(list(ics.values())))
    alive = [k for k, v in ics.items() if np.sign(v) == dominant and abs(v) > 0.02]
    detail = ", ".join(f"{k} {v:+.2f}" for k, v in ics.items())
    if len(alive) <= 1:
        return Attack("regime", False, 0.6,
                      f"Only works in the '{alive[0] if alive else 'no'}' regime ({detail}).",
                      "This is a regime bet, not an edge. Only run it when that regime is confirmed live.")
    return Attack("regime", True, round(1.0 - len(alive) / max(len(ics), 1), 3),
                  f"Present in {len(alive)}/{len(ics)} regimes ({detail}).")


# ── orchestration ─────────────────────────────────────────────────────────────

class RedTeam:
    """Runs every attack against one hypothesis. Deterministic given a seed."""

    def __init__(self, cost_bps: float = 5.0, seed: int = 42, n_permutations: int = 2000) -> None:
        self.cost_bps = cost_bps
        self.seed = seed
        self.n_permutations = n_permutations

    def run(
        self,
        hypothesis: Hypothesis,
        feature_values: Sequence[float],
        forward_returns: Sequence[float],
    ) -> RedTeamReport:
        x = np.asarray(feature_values, dtype=float)
        y = np.asarray(forward_returns, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        ic = rank_ic(x, y)

        report = RedTeamReport(hypothesis_id=hypothesis.id)
        report.attacks = [
            attack_out_of_sample(x, y),
            attack_stability(x, y),
            attack_multiple_testing(ic, len(x), hypothesis.n_considered),
            attack_permutation(x, y, n_sims=self.n_permutations, seed=self.seed),
            attack_cost(y, hypothesis.direction, ic, cost_bps=self.cost_bps),
            attack_regime(x, y),
        ]
        return report


def _norm_cdf(z: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(float(z) / sqrt(2.0)))


def _rolling_std(v: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(v), np.nan)
    for i in range(window, len(v) + 1):
        out[i - 1] = np.std(v[i - window:i])
    out[:window - 1] = out[window - 1]
    return out


if __name__ == "__main__":
    from core.generator import Generator

    rng = np.random.default_rng(7)
    n = 600
    real = rng.normal(0, 1, n)
    fake = rng.normal(0, 1, n)
    # returns ~1% vol, with a modest real edge buried in it
    fwd = (0.12 * real + rng.normal(0, 1, n)) * 0.01

    gen = Generator(seed=1, min_prior=0.0)
    ideas = gen.generate({"real_signal": real, "lucky_noise": fake}, fwd)
    team = RedTeam(cost_bps=5.0)

    print("STARK MK18 - red team demo\n")
    for h in ideas:
        rep = team.run(h, {"real_signal": real, "lucky_noise": fake}[h.feature], fwd)
        print(f"{h.id}  {h.feature}  prior {h.prior:.2f}")
        for a in rep.attacks:
            print(a)
        state = "SURVIVED" if rep.survived else f"KILLED by {', '.join(a.name for a in rep.kill_shots)}"
        print(f"  => {state} | confidence {h.prior:.2f} -> {rep.adjusted_confidence(h.prior):.2f}\n")
