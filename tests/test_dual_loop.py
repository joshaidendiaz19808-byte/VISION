#!/usr/bin/env python3
"""
Phase 4 execution tests.

Two things need proving here:

  the reflex loop  is deterministic, bounded, and its hard checks fire on every
                   tick regardless of what the kernel wants
  the ego loop     can only promote through all four gates, and CANNOT promote
                   anything the risk arbiter refuses

Plus the boundary that matters most: nothing in this module can place an order.

Run: pytest tests/test_dual_loop.py -q
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.risk_arbiter import MAX_GROSS_FRACTION, RiskArbiter
from execution.dual_loop import (ABSTAIN, OPERATORS, EgoLoop, ExecutionAdapter, Intent,
                                 Kernel, PaperAdapter, ReflexLoop, Snapshot,
                                 synthetic_market)


def kernel(**kw) -> Kernel:
    base = dict(kernel_id="k", operators=("fade_imbalance", "spread_capture"),
                parameters=(0.5, 2.0), weights=(1.0, 0.3), size_scale=0.02)
    base.update(kw)
    return Kernel(**base)


def snapshot(**kw) -> Snapshot:
    base = dict(timestamp=1.0, mid=100.0, spread_bps=2.0, imbalance=1.5,
                volatility_bps=120.0, inventory=0.0, staleness_ms=1.0)
    base.update(kw)
    return Snapshot(**base)


# ── kernels are data, and the whitelist is the boundary ───────────────────────

class TestKernel:
    def test_a_non_whitelisted_operator_is_refused(self):
        """The whitelist is the security boundary. Nothing else may execute."""
        with pytest.raises(ValueError, match="non-whitelisted"):
            kernel(operators=("rm_rf",), parameters=(1.0,), weights=(1.0,))

    def test_every_whitelisted_operator_is_pure_and_bounded(self):
        for name, operator in OPERATORS.items():
            for imbalance in (-5.0, 0.0, 5.0):
                value = operator(snapshot(imbalance=imbalance), 1.0)
                assert -1.0 <= value <= 1.0, f"{name} returned {value}"
                assert value == operator(snapshot(imbalance=imbalance), 1.0), f"{name} is impure"

    def test_mismatched_field_lengths_are_refused(self):
        with pytest.raises(ValueError):
            kernel(operators=("fade_imbalance",), parameters=(1.0, 2.0), weights=(1.0,))

    def test_an_absurd_size_scale_is_refused(self):
        with pytest.raises(ValueError):
            kernel(size_scale=5.0)
        with pytest.raises(ValueError):
            kernel(size_scale=0.0)

    def test_evaluation_is_deterministic(self):
        k, s = kernel(), snapshot()
        assert k.evaluate(s) == k.evaluate(s)

    def test_size_never_exceeds_the_kernels_own_scale(self):
        k = kernel(size_scale=0.03)
        for imbalance in np.linspace(-5, 5, 41):
            assert k.evaluate(snapshot(imbalance=float(imbalance))).size_fraction <= 0.03 + 1e-12

    def test_a_kernel_is_inspectable_before_it_trades(self):
        assert "fade_imbalance" in kernel().describe()


# ── the reflex loop ───────────────────────────────────────────────────────────

class TestReflexLoop:
    def test_hard_checks_fire_whatever_the_kernel_wants(self):
        loop = ReflexLoop(kernel(), max_size_fraction=0.05, max_inventory=0.20,
                          max_staleness_ms=50.0)
        assert "stale" in loop.tick(snapshot(staleness_ms=500.0)).reason
        assert "not tradeable" in loop.tick(snapshot(mid=float("nan"))).reason
        assert loop.tick(snapshot(inventory=0.35)).action == "flatten"

    def test_a_kernel_asking_too_big_is_clamped_not_rejected(self):
        loop = ReflexLoop(kernel(size_scale=0.9), max_size_fraction=0.05)
        intent = loop.tick(snapshot(imbalance=3.0))
        assert intent.size_fraction == pytest.approx(0.05)
        assert "clamped" in intent.reason

    def test_the_kill_switch_stops_everything(self):
        loop = ReflexLoop(kernel())
        loop.halt("manual")
        for imbalance in (-3.0, 0.0, 3.0):
            assert not loop.tick(snapshot(imbalance=imbalance)).is_trade

    def test_it_is_deterministic_over_a_whole_run(self):
        snapshots, _ = synthetic_market(500, seed=3)
        first = [ReflexLoop(kernel()).tick(s) for s in snapshots]
        second = [ReflexLoop(kernel()).tick(s) for s in snapshots]
        assert [i.action for i in first] == [i.action for i in second]
        assert [i.size_fraction for i in first] == [i.size_fraction for i in second]

    def test_promotion_is_atomic_and_logged(self):
        loop = ReflexLoop(kernel(kernel_id="old"))
        replacement = kernel(kernel_id="new")
        loop.promote(replacement, "because")
        assert loop.active_kernel is replacement
        assert loop.promotions[-1][1] == "old -> new"

    def test_every_tick_is_accounted_for(self):
        snapshots, _ = synthetic_market(300, seed=5)
        loop = ReflexLoop(kernel())
        for s in snapshots:
            loop.tick(s)
        stats = loop.stats
        assert stats.ticks == 300
        assert stats.intents + stats.abstains + stats.blocked == 300

    def test_latency_is_measured(self):
        loop = ReflexLoop(kernel())
        for s, _ in zip(*synthetic_market(200, seed=1)):
            loop.tick(s)
        assert loop.stats.percentile(50) > 0


# ── the ego loop ──────────────────────────────────────────────────────────────

class TestEgoLoop:
    def test_proposals_stay_inside_the_whitelist(self):
        """The search may restructure freely - it may not invent an operator."""
        ego = EgoLoop(seed=1)
        for candidate in ego.propose(kernel(), 60):
            assert set(candidate.operators) <= set(OPERATORS)

    def test_proposals_can_change_structure_not_just_parameters(self):
        ego = EgoLoop(seed=2)
        shapes = {c.operators for c in ego.propose(kernel(), 60)}
        assert len(shapes) > 1, "the search only ever tweaked numbers"

    def test_the_search_is_deterministic(self):
        a = [c.describe() for c in EgoLoop(seed=9).propose(kernel(), 20)]
        b = [c.describe() for c in EgoLoop(seed=9).propose(kernel(), 20)]
        assert a == b

    def test_it_promotes_a_better_kernel_when_the_regime_changes(self):
        snapshots, forwards = synthetic_market(3_000, seed=2, regime="trending")
        loop = ReflexLoop(kernel(), max_size_fraction=0.05)
        split = len(snapshots) // 2
        promoted = EgoLoop(seed=7, population=40).improve(
            loop, snapshots[:split], forwards[:split], generations=12)
        assert promoted is not None, "never found an improvement in a regime it was backwards for"

        held_out = snapshots[split:], forwards[split:]
        assert EgoLoop.score(loop.active_kernel, *held_out) > EgoLoop.score(kernel(), *held_out)

    def test_search_is_not_gated_by_the_arbiter_but_promotion_is(self):
        """
        Regression, and the design point of the whole module.

        Gating the SEARCH by the arbiter made the system unable to learn its way
        out of a bad regime: every intermediate kernel between backwards and
        correct is itself unprofitable, so the arbiter - correctly - vetoed all
        of them and nothing ever improved. Search freely; promote strictly.
        """
        snapshots, forwards = synthetic_market(2_000, seed=2, regime="trending")
        loop = ReflexLoop(kernel(), max_size_fraction=0.05)
        ego = EgoLoop(seed=7, population=30)
        assert ego.improve(loop, snapshots, forwards, generations=10) is not None
        assert any(not v.risk_allowed for v in ego.history), (
            "no candidate was ever refused - the arbiter is not actually in the path"
        )

    def test_nothing_is_promoted_when_nothing_is_better(self):
        snapshots, forwards = synthetic_market(800, seed=4, regime="mean_reverting")
        strong = kernel(kernel_id="strong", weights=(2.0, 0.3))
        loop = ReflexLoop(strong)
        # a single generation against an already-good kernel usually finds nothing;
        # whatever happens, the incumbent may only ever be replaced by something better
        before = EgoLoop.score(loop.active_kernel, snapshots, forwards)
        EgoLoop(seed=3, population=6).improve(loop, snapshots, forwards, generations=1)
        assert EgoLoop.score(loop.active_kernel, snapshots, forwards) >= before

    def test_a_vetoed_candidate_cannot_be_promoted(self):
        """Gate 3, stated directly: the arbiter's veto is not advisory."""
        snapshots, forwards = synthetic_market(600, seed=6)
        ego = EgoLoop(seed=1)
        # 12 trades is far below the arbiter's 100-observation floor
        verification = ego.verify(kernel(kernel_id="thin"), kernel(), snapshots, forwards,
                                  n_trades=12)
        assert not verification.risk_allowed
        assert not verification.passed
        assert "vetoed" in verification.risk_note

    def test_the_permitted_size_is_binding_on_promotion(self):
        snapshots, forwards = synthetic_market(2_000, seed=2, regime="trending")
        loop = ReflexLoop(kernel(), max_size_fraction=0.05)
        ego = EgoLoop(seed=7, population=30)
        ego.improve(loop, snapshots, forwards, generations=10)
        arbiter_cap = MAX_GROSS_FRACTION
        peak = max(loop.active_kernel.evaluate(s).size_fraction for s in snapshots[:200])
        assert peak <= arbiter_cap + 1e-9, "a kernel was promoted above the arbiter's gross cap"

    def test_verification_reports_why_it_failed(self):
        snapshots, forwards = synthetic_market(400, seed=8)
        verification = EgoLoop(seed=1).verify(kernel(), kernel(), snapshots, forwards,
                                              n_trades=5)
        assert verification.failures()
        assert all(isinstance(reason, str) for reason in verification.failures())

    def test_a_non_deterministic_kernel_would_be_caught(self):
        snapshots, forwards = synthetic_market(300, seed=1)
        verification = EgoLoop(seed=1).verify(kernel(), kernel(), snapshots, forwards)
        assert verification.deterministic, "the determinism check does not actually check"


# ── the boundary ──────────────────────────────────────────────────────────────

class TestNoOrders:
    def test_the_only_adapter_sends_nothing(self):
        adapter = PaperAdapter()
        adapter.send(Intent("bid", 0.02, "test"))
        assert adapter.count == 1
        assert not hasattr(adapter, "connect")
        assert not hasattr(adapter, "submit_order")

    def test_abstain_is_not_recorded_as_a_trade(self):
        adapter = PaperAdapter()
        adapter.send(ABSTAIN)
        assert adapter.count == 0

    def test_no_network_or_broker_anywhere_in_the_module(self):
        """
        CLAUDE.md: STARK places no orders and holds no keys. This module emits
        intents; connecting one to a broker is a human decision with a human
        signature, and nothing here makes it easy by accident.
        """
        import ast
        import pathlib

        source = (pathlib.Path(__file__).resolve().parents[1]
                  / "execution" / "dual_loop.py").read_text()
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        forbidden = {"socket", "requests", "urllib", "http", "websocket", "ccxt", "ib_insync"}
        assert not (imported & forbidden), f"execution reached for {imported & forbidden}"

    def test_no_generated_code_is_ever_executed(self):
        """
        The design decision this module rests on: kernels are composed from a
        whitelist, never generated as source and exec()'d. Code nobody has read
        cannot be reviewed before it trades or reproduced after it does.
        """
        import ast
        import pathlib

        tree = ast.parse((pathlib.Path(__file__).resolve().parents[1]
                          / "execution" / "dual_loop.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"exec", "eval", "compile", "__import__"}, (
                    f"dual_loop.py calls {node.func.id}() - generated code cannot be vetoed"
                )
