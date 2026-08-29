#!/usr/bin/env python3
"""
Phase 1 guardrail tests: does the arbiter actually reject what it must?

These tests exist to fail. Every one of them describes a proposal that a
plausible, well-meaning, confident upstream agent might make, and asserts that
the deterministic layer refuses it anyway. If someone relaxes a limit to make a
strategy fit, a test here turns red and names the limit they moved.

Structure:
  1. the pure math      - closed-form, checked against hand-computed values
  2. hard vetoes        - proposals that must be refused outright
  3. clamps             - proposals that are permitted, but smaller
  4. invariants         - properties that must hold for ANY proposal
  5. determinism/purity - no randomness, no state, no LLM

Run: pytest stark-core/tests/test_risk_arbiter.py -q
"""
from __future__ import annotations

import itertools
import pathlib

import numpy as np
import pytest

from core.risk_arbiter import (
    BPS,
    MAX_DRAWDOWN_FRACTION,
    MAX_GROSS_FRACTION,
    MAX_KELLY_FRACTION,
    MAX_PORTFOLIO_VOL_BPS,
    MAX_POSITION_VOL_BPS,
    MIN_CONFIDENCE_FOR_LIVE,
    MIN_NET_EDGE_BPS,
    MIN_TRADES_FOR_LIVE,
    TARGET_VOL_BPS,
    Proposal,
    RiskArbiter,
    RiskVeto,
    drawdown_scalar,
    kelly_fraction,
    max_weight_for_vol_budget,
    portfolio_volatility,
    variance_contribution,
    vol_target_fraction,
)


@pytest.fixture
def arbiter() -> RiskArbiter:
    return RiskArbiter()


def valid(**overrides) -> Proposal:
    """A proposal that passes everything. Tests break ONE thing at a time."""
    base = dict(
        hypothesis_id="H-VALID",
        size_fraction=0.05,
        confidence=0.45,
        expected_edge_bps=20.0,
        n_trades=1_000,
        survived_red_team=True,
        correlation_to_book=0.1,
        current_drawdown=0.0,
        is_simulated=False,
        volatility_bps=120.0,
        book_volatility_bps=80.0,
        book_weight=0.20,
    )
    base.update(overrides)
    return Proposal(**base)


# ── 1. the pure math ──────────────────────────────────────────────────────────

class TestMath:
    def test_kelly_is_edge_over_variance(self):
        # 50 bps edge, 100 bps vol -> 0.005 / 0.01^2 = 50.0
        assert kelly_fraction(50.0, 100.0) == pytest.approx(50.0)

    def test_kelly_falls_as_volatility_rises(self):
        assert kelly_fraction(20.0, 50.0) > kelly_fraction(20.0, 100.0) > kelly_fraction(20.0, 400.0)

    @pytest.mark.parametrize("vol", [0.0, -10.0, float("nan"), float("inf")])
    def test_kelly_refuses_unmeasurable_volatility(self, vol):
        assert kelly_fraction(20.0, vol) == 0.0

    def test_vol_target_halves_when_volatility_doubles(self):
        assert vol_target_fraction(100.0) == pytest.approx(TARGET_VOL_BPS / 100.0)
        assert vol_target_fraction(200.0) == pytest.approx(vol_target_fraction(100.0) / 2)

    def test_portfolio_vol_matches_the_closed_form(self):
        # w=(0.5, 0.5), sigma=(100, 200) bps, rho=0.5
        got = portfolio_volatility(0.5, 200.0, 0.5, 100.0, 0.5)
        want = np.sqrt(0.25 * (100 * BPS) ** 2 + 0.25 * (200 * BPS) ** 2
                       + 2 * 0.5 * 0.5 * 0.5 * (100 * BPS) * (200 * BPS)) / BPS
        assert got == pytest.approx(want)

    def test_correlation_of_one_is_the_worst_case(self):
        vols = [portfolio_volatility(0.3, 150.0, 0.4, 120.0, rho) for rho in (-1.0, 0.0, 0.5, 1.0)]
        assert vols == sorted(vols), "portfolio vol must rise monotonically with correlation"

    def test_variance_budget_solution_lands_exactly_on_the_budget(self):
        """The quadratic is solved, not searched - the answer must be exact."""
        w = max_weight_for_vol_budget(200.0, 0.75, 190.0, 0.5)
        assert portfolio_volatility(w, 200.0, 0.75, 190.0, 0.5) == pytest.approx(
            MAX_PORTFOLIO_VOL_BPS, rel=1e-9
        )

    def test_variance_budget_is_zero_when_the_book_is_already_over(self):
        assert max_weight_for_vol_budget(100.0, 1.0, MAX_PORTFOLIO_VOL_BPS * 2, 0.0) == 0.0

    def test_variance_contribution_is_a_share(self):
        share = variance_contribution(0.5, 200.0, 0.5, 100.0, 0.3)
        assert 0.0 <= share <= 1.0

    def test_drawdown_scalar_runs_from_one_to_zero(self):
        assert drawdown_scalar(0.0) == 1.0
        assert drawdown_scalar(MAX_DRAWDOWN_FRACTION) == 0.0
        assert drawdown_scalar(MAX_DRAWDOWN_FRACTION * 2) == 0.0     # never negative
        assert drawdown_scalar(MAX_DRAWDOWN_FRACTION / 2) == pytest.approx(0.5)


# ── 2. hard vetoes: proposals that must be refused outright ───────────────────

class TestVetoes:
    """Each case is a different way an upstream agent might be wrong."""

    @pytest.mark.parametrize(
        "label, proposal",
        [
            ("red team killed it", valid(survived_red_team=False)),
            ("sample is an anecdote", valid(n_trades=MIN_TRADES_FOR_LIVE - 1)),
            ("edge under the cost floor", valid(expected_edge_bps=MIN_NET_EDGE_BPS - 0.1)),
            ("negative edge", valid(expected_edge_bps=-40.0)),
            ("evidence is simulated", valid(is_simulated=True)),
            ("volatility is zero", valid(volatility_bps=0.0)),
            ("volatility is NaN", valid(volatility_bps=float("nan"))),
            ("volatility is infinite", valid(volatility_bps=float("inf"))),
            ("strategy too wild to size", valid(volatility_bps=MAX_POSITION_VOL_BPS + 1)),
            ("book already over its variance budget",
             valid(book_weight=1.0, book_volatility_bps=MAX_PORTFOLIO_VOL_BPS + 1)),
            ("drawdown halt is binding", valid(current_drawdown=MAX_DRAWDOWN_FRACTION)),
            ("confidence below the floor", valid(confidence=MIN_CONFIDENCE_FOR_LIVE - 0.001)),
        ],
    )
    def test_invalid_proposal_is_vetoed(self, arbiter, label, proposal):
        ruling = arbiter.rule(proposal)
        assert ruling.vetoed, f"{label} was NOT vetoed: {ruling}"
        assert ruling.max_size_fraction == 0.0, f"{label} was vetoed but still got size"

    def test_veto_raises_and_cannot_be_talked_out_of(self, arbiter):
        ruling = arbiter.rule(valid(n_trades=10))
        with pytest.raises(RiskVeto) as excinfo:
            ruling.enforce()
        assert "VETOED" in str(excinfo.value)
        assert "10 observations" in str(excinfo.value)   # it says WHY

    def test_a_veto_names_a_hard_violation(self, arbiter):
        ruling = arbiter.rule(valid(is_simulated=True))
        assert any(v.hard for v in ruling.violations)
        assert all(v.fix for v in ruling.violations), "every violation must say how to fix it"

    def test_maximum_confidence_cannot_buy_a_veto_off(self, arbiter):
        """The neural half gets no vote. This is the whole thesis of the module."""
        for conf in (0.9, 0.99, 1.0, 5.0):
            ruling = arbiter.rule(valid(confidence=conf, n_trades=3))
            assert ruling.vetoed
            with pytest.raises(RiskVeto):
                ruling.enforce()

    def test_an_enormous_edge_cannot_buy_a_veto_off(self, arbiter):
        ruling = arbiter.rule(valid(expected_edge_bps=100_000.0, n_trades=5))
        assert ruling.vetoed

    def test_a_valid_proposal_is_not_vetoed(self, arbiter):
        """Guard against a suite that passes by rejecting everything."""
        ruling = arbiter.rule(valid())
        assert not ruling.vetoed
        assert ruling.enforce() > 0.0


# ── 3. clamps: permitted, but smaller ─────────────────────────────────────────

class TestClamps:
    def test_greed_is_clamped_to_the_gross_cap(self, arbiter):
        ruling = arbiter.rule(valid(size_fraction=0.90, confidence=1.0,
                                    volatility_bps=20.0, book_weight=0.0,
                                    book_volatility_bps=0.0))
        assert ruling.decision == "CLAMP"
        assert ruling.max_size_fraction <= MAX_GROSS_FRACTION

    def test_quarter_kelly_binds_on_a_thin_edge_in_a_wild_strategy(self, arbiter):
        p = valid(size_fraction=0.20, expected_edge_bps=6.0, volatility_bps=800.0,
                  book_weight=0.0, book_volatility_bps=0.0)
        ruling = arbiter.rule(p)
        cap = MAX_KELLY_FRACTION * kelly_fraction(6.0, 800.0)
        assert ruling.binding_constraint == "kelly-cap"
        assert cap - 1e-4 <= ruling.max_size_fraction <= cap, "size is floored to the cap, never above it"

    def test_variance_budget_binds_on_a_loaded_book(self, arbiter):
        p = valid(size_fraction=0.25, expected_edge_bps=30.0, confidence=0.7,
                  volatility_bps=200.0, book_volatility_bps=190.0,
                  book_weight=0.75, correlation_to_book=0.5)
        ruling = arbiter.rule(p)
        assert ruling.binding_constraint == "variance-budget"
        assert ruling.projected_vol_bps <= MAX_PORTFOLIO_VOL_BPS + 1e-6

    def test_drawdown_scales_size_down_linearly(self, arbiter):
        flat = arbiter.rule(valid(current_drawdown=0.0)).max_size_fraction
        half = arbiter.rule(valid(current_drawdown=MAX_DRAWDOWN_FRACTION / 2)).max_size_fraction
        assert half == pytest.approx(flat * 0.5, rel=1e-6)

    def test_correlated_ideas_are_not_new_bets(self, arbiter):
        low = arbiter.rule(valid(correlation_to_book=0.1)).max_size_fraction
        high = arbiter.rule(valid(correlation_to_book=0.95)).max_size_fraction
        assert high < low

    def test_granted_size_is_floored_never_rounded(self, arbiter):
        """
        Regression: size was rounded to 4dp, which rounded a 0.070080 variance
        solution UP to 0.0701 and breached the budget. A risk cap may only ever
        be approached from below.
        """
        p = valid(size_fraction=0.25, expected_edge_bps=30.0, confidence=0.7,
                  volatility_bps=200.0, book_volatility_bps=190.0,
                  book_weight=0.75, correlation_to_book=0.5)
        ruling = arbiter.rule(p)
        assert ruling.max_size_fraction <= ruling.caps["variance-budget"]
        assert ruling.projected_vol_bps <= MAX_PORTFOLIO_VOL_BPS

    def test_the_ruling_names_the_binding_constraint(self, arbiter):
        ruling = arbiter.rule(valid(size_fraction=0.90, confidence=1.0,
                                    volatility_bps=20.0, book_weight=0.0,
                                    book_volatility_bps=0.0))
        cap = ruling.caps[ruling.binding_constraint]
        assert ruling.binding_constraint in ruling.caps
        assert cap - 1e-4 <= ruling.max_size_fraction <= cap


# ── 4. invariants that must hold for ANY proposal ─────────────────────────────

GRID = list(itertools.product(
    [0.0, 0.05, 0.5, 5.0],            # size_fraction
    [0.0, 0.06, 0.5, 1.0],            # confidence
    [-5.0, 2.5, 40.0, 500.0],         # edge_bps
    [50, 1_000],                      # n_trades
    [10.0, 120.0, 900.0],             # volatility_bps
    [0.0, 0.19],                      # drawdown
))


@pytest.mark.parametrize("size, conf, edge, n, vol, dd", GRID)
def test_size_is_never_negative_nan_or_over_the_cap(arbiter, size, conf, edge, n, vol, dd):
    """384 proposals, including nonsense ones. No path may produce an invalid size."""
    ruling = arbiter.rule(valid(size_fraction=size, confidence=conf, expected_edge_bps=edge,
                                n_trades=n, volatility_bps=vol, current_drawdown=dd))
    got = ruling.max_size_fraction
    assert np.isfinite(got)
    assert 0.0 <= got <= MAX_GROSS_FRACTION
    assert got <= size + 1e-9, "the arbiter may only ever grant LESS than requested"


@pytest.mark.parametrize("size, conf, edge, n, vol, dd", GRID)
def test_permitted_size_never_breaches_the_variance_budget(arbiter, size, conf, edge, n, vol, dd):
    p = valid(size_fraction=size, confidence=conf, expected_edge_bps=edge, n_trades=n,
              volatility_bps=vol, current_drawdown=dd,
              book_weight=0.4, book_volatility_bps=120.0, correlation_to_book=0.4)
    ruling = arbiter.rule(p)
    projected = portfolio_volatility(ruling.max_size_fraction, vol, 0.4, 120.0, 0.4)
    assert projected <= MAX_PORTFOLIO_VOL_BPS + 1e-6


def test_more_risk_never_earns_more_size(arbiter):
    """Monotonicity: raising volatility can only shrink the permitted position."""
    sizes = [
        arbiter.rule(valid(size_fraction=0.25, confidence=1.0, expected_edge_bps=50.0,
                           volatility_bps=v)).max_size_fraction
        for v in (50.0, 100.0, 200.0, 400.0, 800.0)
    ]
    assert sizes == sorted(sizes, reverse=True), sizes


# ── 5. determinism and purity ─────────────────────────────────────────────────

class TestPurity:
    def test_identical_proposals_give_identical_rulings(self, arbiter):
        p = valid()
        first, second = arbiter.rule(p), arbiter.rule(p)
        assert first.max_size_fraction == second.max_size_fraction
        assert first.decision == second.decision
        assert first.binding_constraint == second.binding_constraint

    def test_the_arbiter_is_stateless(self, arbiter):
        """A veto must not linger and poison the next proposal, or vice versa."""
        clean_first = arbiter.rule(valid()).max_size_fraction
        arbiter.rule(valid(n_trades=1))          # a veto in between
        arbiter.rule(valid(is_simulated=True))   # and another
        assert arbiter.rule(valid()).max_size_fraction == clean_first

    def test_no_llm_and_no_randomness_in_the_module(self):
        """
        CLAUDE.md rule 2: absolute zero LLM logic, absolute zero sampling.

        Parsed with ast rather than grepped, so the module's own prose about
        having no LLM does not trip a search for the word "LLM". Imports are
        allowlisted: anything not on this list has to be argued for.
        """
        import ast

        path = pathlib.Path(__file__).resolve().parents[1] / "core" / "risk_arbiter.py"
        tree = ast.parse(path.read_text())

        allowed = {"__future__", "dataclasses", "typing", "numpy"}
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported <= allowed, f"risk_arbiter.py imports {imported - allowed}"

        # no sampling: nothing may reference a random source in executable code
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert "random" not in node.attr.lower()
            if isinstance(node, ast.Name):
                assert "random" not in node.id.lower()

    def test_the_arbiter_imports_nothing_from_stark(self):
        src = (pathlib.Path(__file__).resolve().parents[1] / "core" / "risk_arbiter.py").read_text()
        for pkg in ("core.", "simulation.", "tensors.", "interface."):
            assert f"import {pkg}" not in src and f"from {pkg}" not in src
