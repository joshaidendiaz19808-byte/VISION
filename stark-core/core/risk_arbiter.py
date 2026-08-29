#!/usr/bin/env python3
"""
Level 1 / Phase 1 - Neuro-symbolic hard guardrails. The Cognitive Nucleus's veto.

This module is the adult in the room. Everything upstream of it is allowed to be
clever; nothing downstream of it is allowed to be.

ZERO LLM LOGIC. There is no model, no prompt, no learned threshold and no
sampling anywhere in this file. Every limit is a hard-coded constant and every
decision is closed-form NumPy arithmetic on those constants. Given the same
Proposal it returns the same Ruling, on any machine, forever. That property is
the entire point: an LLM may propose, but only this file may permit.

Design constraints (CLAUDE.md rules 1 and 2):
  * It imports nothing from STARK. It takes plain floats and ints, so no caller
    can hand it a clever object with a helpful __bool__ and walk through the wall.
  * Every rule is one readable predicate. A human must be able to audit the whole
    ruleset in a minute.
  * A VETO is final. `RiskVeto` is raised by enforce() and must never be caught
    and downgraded - catch it to report it, then stop.

Three families of limit, in the order the spec names them:

  position size   gross cap, confidence scaling, quarter-Kelly
  variance        vol targeting + a portfolio variance budget solved exactly
  drawdown        a hard halt, and linear de-risking on the way down

The "neuro" half of neuro-symbolic is a single scalar input: `confidence`, the
red team's surviving number. The symbolic half - this file - decides what that
confidence is permitted to buy. It is a one-way valve.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ── THE HARD LIMITS ───────────────────────────────────────────────────────────
# The only place in STARK where a risk boundary may be defined. Change a number
# here and the entire system's behaviour changes; change one anywhere else and
# you have introduced a bug.

# -- position size --
MAX_GROSS_FRACTION = 0.25        # no single idea gets more than 25% of the book
MAX_KELLY_FRACTION = 0.25        # quarter-Kelly. Full Kelly is a theoretical
                                 # optimum for people who cannot be ruined.
MIN_CONFIDENCE_FOR_LIVE = 0.05

# -- variance (all volatilities are per-period, in bps) --
TARGET_VOL_BPS = 75.0            # what one position is sized to contribute
MAX_PORTFOLIO_VOL_BPS = 150.0    # total book variance budget - a hard ceiling
MAX_POSITION_VOL_BPS = 1_000.0   # a strategy wilder than 10%/period is not sized

# -- evidence --
MIN_TRADES_FOR_LIVE = 100        # below this a track record is an anecdote
MIN_NET_EDGE_BPS = 2.0           # thinner than this and slippage owns your P&L

# -- drawdown --
MAX_DRAWDOWN_FRACTION = 0.20     # stop the whole book here
MAX_CORRELATION = 0.70           # to the existing book: past this it is one bet

BPS = 1e-4


class RiskVeto(Exception):
    """Raised by Ruling.enforce(). Never catch this to continue - only to report."""


# ── pure math. No state, no I/O, no randomness. ───────────────────────────────

def kelly_fraction(edge_bps: float, vol_bps: float) -> float:
    """
    f* = edge / variance, the growth-optimal fraction.

    Kelly is famously aggressive - it is optimal only if your edge estimate is
    exact, which it never is. We take a quarter of it (MAX_KELLY_FRACTION) and
    still treat that as an upper bound rather than a target.
    """
    edge, vol = float(edge_bps) * BPS, float(vol_bps) * BPS
    if not np.isfinite(edge) or not np.isfinite(vol) or vol <= 0:
        return 0.0
    return float(max(edge / (vol * vol), 0.0))


def vol_target_fraction(vol_bps: float, target_bps: float = TARGET_VOL_BPS) -> float:
    """Size so this position contributes `target_bps` of volatility. w = target / sigma."""
    if not np.isfinite(vol_bps) or vol_bps <= 0:
        return 0.0
    return float(target_bps / vol_bps)


def covariance_matrix(vols_bps: np.ndarray, correlation: float) -> np.ndarray:
    """Two-asset covariance from volatilities and one correlation. Exact, not fitted."""
    v = np.asarray(vols_bps, dtype=float) * BPS
    corr = np.array([[1.0, float(correlation)], [float(correlation), 1.0]])
    return np.outer(v, v) * corr


def portfolio_volatility(
    weight_new: float, vol_new_bps: float,
    weight_book: float, vol_book_bps: float,
    correlation: float,
) -> float:
    """sqrt(w' C w), in bps. The number the variance budget is spent against."""
    w = np.array([float(weight_book), float(weight_new)], dtype=float)
    cov = covariance_matrix(np.array([vol_book_bps, vol_new_bps]), correlation)
    var = float(w @ cov @ w)
    return float(np.sqrt(max(var, 0.0)) / BPS)


def max_weight_for_vol_budget(
    vol_new_bps: float, weight_book: float, vol_book_bps: float,
    correlation: float, budget_bps: float = MAX_PORTFOLIO_VOL_BPS,
) -> float:
    """
    Largest new weight that keeps total book volatility <= budget.

    Solved exactly, not searched: portfolio variance is a quadratic in the new
    weight, so a w^2 + b w + c = 0 with

        a = sigma_new^2
        b = 2 * w_book * rho * sigma_book * sigma_new
        c = (w_book * sigma_book)^2 - budget^2

    A negative discriminant or c > 0 means the book is already at or over budget
    with zero new risk, so the answer is zero. No new position, no exceptions.
    """
    s_new, s_book = float(vol_new_bps) * BPS, float(vol_book_bps) * BPS
    w_book, budget = float(weight_book), float(budget_bps) * BPS
    if not np.isfinite(s_new) or s_new <= 0:
        return 0.0
    a = s_new ** 2
    b = 2.0 * w_book * float(correlation) * s_book * s_new
    c = (w_book * s_book) ** 2 - budget ** 2
    if c > 0:
        return 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return 0.0
    return float(max((-b + np.sqrt(disc)) / (2.0 * a), 0.0))


def variance_contribution(
    weight_new: float, vol_new_bps: float,
    weight_book: float, vol_book_bps: float,
    correlation: float,
) -> float:
    """
    This position's share of total portfolio variance, in [0, 1].

    Marginal contribution: w_i (C w)_i / (w' C w). What the position actually
    costs the book, not what it costs standalone.
    """
    w = np.array([float(weight_book), float(weight_new)], dtype=float)
    cov = covariance_matrix(np.array([vol_book_bps, vol_new_bps]), correlation)
    total = float(w @ cov @ w)
    if total <= 0:
        return 0.0
    return float(w[1] * (cov @ w)[1] / total)


def drawdown_scalar(drawdown: float, halt: float = MAX_DRAWDOWN_FRACTION) -> float:
    """Linear de-risking: full size at the high-water mark, zero at the halt."""
    return float(np.clip(1.0 - float(drawdown) / halt, 0.0, 1.0))


# ── the objects that cross the boundary ───────────────────────────────────────

@dataclass(frozen=True)
class Proposal:
    """A request to risk money. Plain primitives only, by design."""

    hypothesis_id: str
    size_fraction: float           # requested fraction of the book
    confidence: float              # red team's post-attack number, [0, 1]
    expected_edge_bps: float       # est. NET of costs
    n_trades: int                  # observations behind the claim
    survived_red_team: bool
    correlation_to_book: float = 0.0
    current_drawdown: float = 0.0  # 0.08 == book is 8% off its high
    is_simulated: bool = False     # simulated evidence never counts as live evidence
    volatility_bps: float = 100.0  # per-period sigma of THIS strategy's returns
    book_volatility_bps: float = 0.0
    book_weight: float = 0.0       # gross weight already deployed


@dataclass(frozen=True)
class Violation:
    rule: str
    hard: bool
    message: str
    fix: str

    def __str__(self) -> str:
        return f"  [{'VETO' if self.hard else 'CLAMP'}] {self.rule}: {self.message}"


@dataclass
class Ruling:
    decision: str                       # "ALLOW" | "CLAMP" | "VETO"
    max_size_fraction: float
    violations: list[Violation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    binding_constraint: str = "requested"
    caps: dict[str, float] = field(default_factory=dict)
    projected_vol_bps: float = 0.0

    @property
    def vetoed(self) -> bool:
        return self.decision == "VETO"

    def enforce(self) -> float:
        """Return the permitted size, or raise. The only sanctioned exit."""
        if self.vetoed:
            reasons = "; ".join(v.message for v in self.violations if v.hard)
            raise RiskVeto(f"VETOED: {reasons}")
        return self.max_size_fraction

    def fixes(self) -> list[str]:
        return [v.fix for v in self.violations]

    def __str__(self) -> str:
        head = f"{self.decision} - max size {self.max_size_fraction:.3f} of book"
        if not self.vetoed:
            head += f" (bound by {self.binding_constraint}, book vol {self.projected_vol_bps:.0f} bps)"
        lines = [str(v) for v in self.violations] + [f"  note: {n}" for n in self.notes]
        return head + ("\n" + "\n".join(lines) if lines else "")


# ── the ruleset ───────────────────────────────────────────────────────────────
# (name, hard?, predicate -> True means VIOLATED, message, fix)
Rule = tuple[str, bool, Callable[[Proposal], bool], Callable[[Proposal], str], str]

RULES: list[Rule] = [
    (
        "red-team-kill", True,
        lambda p: not p.survived_red_team,
        lambda p: "The red team killed this hypothesis.",
        "A dead hypothesis gets zero size. Fix the finding or drop the idea.",
    ),
    (
        "sample-size", True,
        lambda p: p.n_trades < MIN_TRADES_FOR_LIVE,
        lambda p: f"Only {p.n_trades} observations; {MIN_TRADES_FOR_LIVE} is the floor for live risk.",
        "Paper trade until the sample is real. A winning run under 100 trades is luck.",
    ),
    (
        "edge-vs-cost", True,
        lambda p: p.expected_edge_bps < MIN_NET_EDGE_BPS,
        lambda p: f"Est. net edge {p.expected_edge_bps:.1f} bps is under the {MIN_NET_EDGE_BPS:.0f} bps floor.",
        "Trade less often or hold longer so the edge clears the toll.",
    ),
    (
        "simulated-evidence", True,
        lambda p: p.is_simulated,
        lambda p: "Evidence is simulated. Simulation sizes nothing.",
        "Re-run the claim on out-of-sample real data before requesting size.",
    ),
    (
        "unmeasurable-variance", True,
        lambda p: not np.isfinite(p.volatility_bps) or p.volatility_bps <= 0,
        lambda p: f"Volatility is {p.volatility_bps} - unmeasurable, so unsizeable.",
        "You cannot budget variance you cannot measure. Estimate sigma first.",
    ),
    (
        "position-vol-ceiling", True,
        lambda p: p.volatility_bps > MAX_POSITION_VOL_BPS,
        lambda p: f"Strategy vol {p.volatility_bps:.0f} bps exceeds the "
                  f"{MAX_POSITION_VOL_BPS:.0f} bps ceiling for a single position.",
        "Too wild to size at any weight. Damp the strategy or trade it smaller elsewhere.",
    ),
    (
        "book-over-variance-budget", True,
        lambda p: abs(p.book_weight) * p.book_volatility_bps > MAX_PORTFOLIO_VOL_BPS,
        lambda p: f"Book already runs {abs(p.book_weight) * p.book_volatility_bps:.0f} bps vol "
                  f"against a {MAX_PORTFOLIO_VOL_BPS:.0f} bps budget.",
        "The budget is spent. Reduce existing risk before adding any.",
    ),
    (
        "drawdown-halt", True,
        lambda p: p.current_drawdown >= MAX_DRAWDOWN_FRACTION,
        lambda p: f"Book is {p.current_drawdown*100:.0f}% off its high; the "
                  f"{MAX_DRAWDOWN_FRACTION*100:.0f}% halt is binding.",
        "Stop adding risk. Reduce, then review the whole book before re-entering.",
    ),
    (
        "min-confidence", True,
        lambda p: p.confidence < MIN_CONFIDENCE_FOR_LIVE,
        lambda p: f"Post-attack confidence {p.confidence:.3f} is below the {MIN_CONFIDENCE_FOR_LIVE} floor.",
        "Not enough conviction survived the red team to justify any size.",
    ),
    (
        "gross-cap", False,
        lambda p: p.size_fraction > MAX_GROSS_FRACTION,
        lambda p: f"Requested {p.size_fraction:.2f} of book; the cap is {MAX_GROSS_FRACTION:.2f}.",
        "Clamped to the cap. No single idea gets to be the whole book.",
    ),
    (
        "correlation", False,
        lambda p: p.correlation_to_book > MAX_CORRELATION,
        lambda p: f"{p.correlation_to_book:.2f} correlated to the existing book - this is not a new bet.",
        "Clamped. Diversify the driver or replace the existing position instead of doubling it.",
    ),
    (
        "confidence-scaling", False,
        lambda p: p.size_fraction > p.confidence * MAX_GROSS_FRACTION * 2,
        lambda p: f"Size {p.size_fraction:.2f} outruns confidence {p.confidence:.2f}.",
        "Clamped to what the evidence supports. Size follows conviction, not hope.",
    ),
    (
        "kelly-cap", False,
        lambda p: p.size_fraction > MAX_KELLY_FRACTION * kelly_fraction(
            p.expected_edge_bps, p.volatility_bps),
        lambda p: f"Size {p.size_fraction:.3f} exceeds quarter-Kelly "
                  f"{MAX_KELLY_FRACTION * kelly_fraction(p.expected_edge_bps, p.volatility_bps):.3f} "
                  f"for a {p.expected_edge_bps:.1f} bps edge at {p.volatility_bps:.0f} bps vol.",
        "Clamped. Kelly assumes your edge estimate is exact; it is not.",
    ),
    (
        "variance-budget", False,
        lambda p: p.size_fraction > max_weight_for_vol_budget(
            p.volatility_bps, p.book_weight, p.book_volatility_bps, p.correlation_to_book),
        lambda p: f"Size {p.size_fraction:.3f} would push book vol past the "
                  f"{MAX_PORTFOLIO_VOL_BPS:.0f} bps budget.",
        "Clamped to the variance budget. The budget is the constraint, not the request.",
    ),
]


class RiskArbiter:
    """
    Applies the ruleset. Stateless, deterministic, boring on purpose.

    The permitted size is the MINIMUM of every cap. Not a weighted blend, not a
    negotiation - the tightest constraint wins outright, and the ruling names it.
    """

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules = rules if rules is not None else RULES

    def caps(self, p: Proposal) -> dict[str, float]:
        """Every ceiling this proposal is subject to, before drawdown scaling."""
        return {
            "requested": float(p.size_fraction),
            "gross-cap": MAX_GROSS_FRACTION,
            "confidence-scaling": float(p.confidence) * MAX_GROSS_FRACTION * 2.0,
            "kelly-cap": MAX_KELLY_FRACTION * kelly_fraction(p.expected_edge_bps, p.volatility_bps),
            "vol-target": vol_target_fraction(p.volatility_bps),
            "variance-budget": max_weight_for_vol_budget(
                p.volatility_bps, p.book_weight, p.book_volatility_bps, p.correlation_to_book),
        }

    def rule(self, proposal: Proposal) -> Ruling:
        violations = [
            Violation(name, hard, msg(proposal), fix)
            for name, hard, pred, msg, fix in self.rules
            if pred(proposal)
        ]
        hard_hits = [v for v in violations if v.hard]
        if hard_hits:
            return Ruling("VETO", 0.0, violations,
                          ["Veto is final. No downstream module may override it."],
                          binding_constraint=hard_hits[0].rule)

        caps = self.caps(proposal)
        binding = min(caps, key=caps.get)
        size = float(caps[binding])

        if proposal.correlation_to_book > MAX_CORRELATION:
            size *= 1.0 - proposal.correlation_to_book
            binding = "correlation"

        scalar = drawdown_scalar(proposal.current_drawdown)
        # FLOOR, never round. Rounding to 4dp can round UP past a hard cap - a
        # 0.070080 variance-budget solution becoming 0.0701 breaches the budget by
        # a hair, and "by a hair" is not a category this module recognises.
        size = float(np.floor(max(size * scalar, 0.0) * 1e4) / 1e4)

        notes: list[str] = []
        if proposal.current_drawdown > 0:
            notes.append(
                f"Size scaled by {scalar:.2f}x: the book is "
                f"{proposal.current_drawdown*100:.0f}% off its high."
            )
            binding = "drawdown" if scalar < 1.0 and size < caps["requested"] else binding

        projected = portfolio_volatility(
            size, proposal.volatility_bps, proposal.book_weight,
            proposal.book_volatility_bps, proposal.correlation_to_book,
        )
        decision = "ALLOW" if not violations and size >= proposal.size_fraction else "CLAMP"
        return Ruling(decision, size, violations, notes, binding, caps, projected)


if __name__ == "__main__":
    arb = RiskArbiter()
    cases = [
        ("clean idea, sensible size",
         Proposal("H-1", 0.05, 0.42, 18.0, 900, True, volatility_bps=120.0)),
        ("greedy size on a decent idea",
         Proposal("H-2", 0.60, 0.55, 22.0, 1200, True, volatility_bps=90.0)),
        ("thin edge, wild strategy (variance binds)",
         Proposal("H-3", 0.20, 0.60, 6.0, 2000, True, volatility_bps=800.0)),
        ("book already carrying most of its variance budget",
         Proposal("H-4", 0.25, 0.70, 30.0, 3000, True, volatility_bps=200.0,
                  book_volatility_bps=190.0, book_weight=0.75, correlation_to_book=0.5)),
        ("volatility unmeasurable",
         Proposal("H-5", 0.05, 0.50, 20.0, 800, True, volatility_bps=0.0)),
        ("thin sample",
         Proposal("H-6", 0.05, 0.40, 20.0, 41, True)),
        ("killed by the red team",
         Proposal("H-7", 0.05, 0.00, 20.0, 5000, False)),
        ("simulation dressed as evidence",
         Proposal("H-8", 0.05, 0.80, 40.0, 4000, True, is_simulated=True)),
        ("book already 15% underwater",
         Proposal("H-9", 0.10, 0.60, 25.0, 2000, True, current_drawdown=0.15)),
    ]

    print("STARK MK18 - risk arbiter demo (Phase 1: the veto)\n")
    for label, p in cases:
        r = arb.rule(p)
        print(f"{label}:")
        print(" ", str(r).replace("\n", "\n "))
        try:
            r.enforce()
        except RiskVeto as e:
            print(f"  enforce() raised -> {e}")
        print()
