#!/usr/bin/env python3
"""
Level 1 - Neuro-symbolic hard guardrails.

This module is the adult in the room. Everything upstream of it is allowed to be
clever; nothing downstream of it is allowed to be.

Design constraints (see CLAUDE.md sections 1 and 2):
  * It imports nothing from STARK. It takes plain numbers so no caller can hand
    it a custom object with a helpful __bool__ and walk through the wall.
  * Every rule is one readable predicate. No learned thresholds. A human must be
    able to audit the whole ruleset in a minute.
  * A VETO is final. `RiskVeto` is raised by enforce() and must never be caught
    and downgraded - catch it to report it, then stop.

The "neuro" half is a single input: `confidence`, the red team's surviving
number. The symbolic half decides what that confidence is permitted to buy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# ── the hard limits. This is the only place in STARK they may be defined. ─────
MAX_GROSS_FRACTION = 0.25      # no single idea gets more than 25% of the book
MIN_TRADES_FOR_LIVE = 100      # below this a track record is an anecdote
MIN_NET_EDGE_BPS = 2.0         # thinner than this and slippage owns your P&L
MAX_DRAWDOWN_FRACTION = 0.20   # stop the whole book here
MAX_CORRELATION = 0.70         # to the existing book: past this it is one bet
MIN_CONFIDENCE_FOR_LIVE = 0.05


class RiskVeto(Exception):
    """Raised by Ruling.enforce(). Never catch this to continue - only to report."""


@dataclass(frozen=True)
class Proposal:
    """A request to risk money. Plain primitives only, by design."""

    hypothesis_id: str
    size_fraction: float          # requested fraction of the book
    confidence: float             # red team's post-attack number, [0, 1]
    expected_edge_bps: float      # est. NET of costs
    n_trades: int                 # observations behind the claim
    survived_red_team: bool
    correlation_to_book: float = 0.0
    current_drawdown: float = 0.0  # 0.08 == book is 8% off its high
    is_simulated: bool = False     # simulated evidence never counts as live evidence


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
        "drawdown-halt", True,
        lambda p: p.current_drawdown >= MAX_DRAWDOWN_FRACTION,
        lambda p: f"Book is {p.current_drawdown*100:.0f}% off its high; the {MAX_DRAWDOWN_FRACTION*100:.0f}% halt is binding.",
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
]


class RiskArbiter:
    """Applies the ruleset. Stateless, deterministic, boring on purpose."""

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules = rules if rules is not None else RULES

    def rule(self, proposal: Proposal) -> Ruling:
        violations = [
            Violation(name, hard, msg(proposal), fix)
            for name, hard, pred, msg, fix in self.rules
            if pred(proposal)
        ]
        hard_hits = [v for v in violations if v.hard]

        if hard_hits:
            return Ruling("VETO", 0.0, violations,
                          ["Veto is final. No downstream module may override it."])

        size = min(
            proposal.size_fraction,
            MAX_GROSS_FRACTION,
            proposal.confidence * MAX_GROSS_FRACTION * 2,
        )
        if proposal.correlation_to_book > MAX_CORRELATION:
            size *= 1.0 - proposal.correlation_to_book
        # scale down as the book bleeds: full size at flat, zero at the halt
        size *= max(0.0, 1.0 - proposal.current_drawdown / MAX_DRAWDOWN_FRACTION)
        size = round(max(size, 0.0), 4)

        notes: list[str] = []
        if proposal.current_drawdown > 0:
            notes.append(
                f"Size scaled by {1 - proposal.current_drawdown / MAX_DRAWDOWN_FRACTION:.2f}x: "
                f"the book is {proposal.current_drawdown*100:.0f}% off its high."
            )
        decision = "ALLOW" if not violations and size >= proposal.size_fraction else "CLAMP"
        return Ruling(decision, size, violations, notes)


if __name__ == "__main__":
    arb = RiskArbiter()
    cases = [
        ("clean idea, sensible size",
         Proposal("H-1", 0.05, 0.42, 18.0, 900, True)),
        ("greedy size on a decent idea",
         Proposal("H-2", 0.60, 0.55, 22.0, 1200, True)),
        ("thin sample",
         Proposal("H-3", 0.05, 0.40, 20.0, 41, True)),
        ("killed by the red team",
         Proposal("H-4", 0.05, 0.00, 20.0, 5000, False)),
        ("simulation dressed as evidence",
         Proposal("H-5", 0.05, 0.80, 40.0, 4000, True, is_simulated=True)),
        ("book already 15% underwater",
         Proposal("H-6", 0.10, 0.60, 25.0, 2000, True, current_drawdown=0.15)),
    ]

    print("STARK MK18 - risk arbiter demo\n")
    for label, p in cases:
        r = arb.rule(p)
        print(f"{label}:")
        print(" ", str(r).replace("\n", "\n "))
        try:
            r.enforce()
        except RiskVeto as e:
            print(f"  enforce() raised -> {e}")
        print()
