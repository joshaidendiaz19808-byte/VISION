#!/usr/bin/env python3
"""
STARK MK18 smoke tests.

Two jobs:
  1. every module imports, runs, and is deterministic
  2. the rules in CLAUDE.md are enforced by code, not by good intentions

Run:  python3 stark-core/tests/test_smoke.py     (or: pytest stark-core/tests)
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.generator import Generator, rank_ic, shrink
from core.red_team import RedTeam, KILL_THRESHOLD
from core.risk_arbiter import (MAX_GROSS_FRACTION, Proposal, RiskArbiter,
                               RiskVeto, MIN_TRADES_FOR_LIVE)
from simulation.active_inference import Action, ActiveInferenceLoop, Observation
from simulation.causal_engine import CausalGraph, Multiverse
from tensors.market_tensors import MarketTensor, synth_panel
from interface.jarvis_node import JarvisNode


def fixture(n: int = 600, strength: float = 0.15, seed: int = 7):
    rng = np.random.default_rng(seed)
    real = rng.normal(0, 1, n)
    features = {"real_signal": real, "pure_noise": rng.normal(0, 1, n)}
    fwd = (strength * real + rng.normal(0, 1, n)) * 0.01
    return features, fwd


# ── level 1 ───────────────────────────────────────────────────────────────────

def test_rank_ic_is_sane():
    x = np.arange(100, dtype=float)
    assert rank_ic(x, x) > 0.99
    assert rank_ic(x, -x) < -0.99
    assert rank_ic(x, np.ones(100)) == 0.0          # a constant cannot correlate


def test_thin_samples_cannot_buy_confidence():
    """CLAUDE.md 3: n=12 is not evidence."""
    assert shrink(0.9, 10) < shrink(0.9, 1000)
    assert shrink(0.9, 10) < 0.2


def test_generator_ranks_signal_above_noise():
    features, fwd = fixture()
    ideas = Generator(seed=1, min_prior=0.0).generate(features, fwd)
    assert ideas and ideas[0].feature == "real_signal"
    assert all(h.n_considered == len(features) for h in ideas)   # search is always charged for


def test_generator_is_deterministic():
    features, fwd = fixture()
    a = Generator(seed=1, min_prior=0.0).generate(features, fwd)
    b = Generator(seed=1, min_prior=0.0).generate(features, fwd)
    assert a == b


def test_red_team_kills_pure_noise():
    features, fwd = fixture()
    gen = Generator(seed=1, min_prior=0.0)
    team = RedTeam(seed=42, n_permutations=500)
    noise = next(h for h in gen.generate(features, fwd) if h.feature == "pure_noise")
    report = team.run(noise, features["pure_noise"], fwd)
    assert not report.survived
    assert report.adjusted_confidence(noise.prior) == 0.0


def test_red_team_spares_a_real_edge():
    features, fwd = fixture(n=1200, strength=0.25)
    gen = Generator(seed=1, min_prior=0.0)
    team = RedTeam(seed=42, n_permutations=500)
    real = next(h for h in gen.generate(features, fwd) if h.feature == "real_signal")
    report = team.run(real, features["real_signal"], fwd)
    assert report.survived, [str(a) for a in report.attacks]


def test_confidence_only_ever_falls():
    """CLAUDE.md 7: nothing downstream of the red team talks conviction back up."""
    features, fwd = fixture(n=1200, strength=0.25)
    gen = Generator(seed=1, min_prior=0.0)
    team = RedTeam(seed=42, n_permutations=300)
    for h in gen.generate(features, fwd):
        report = team.run(h, features[h.feature], fwd)
        assert report.adjusted_confidence(h.prior) <= h.prior + 1e-9


def test_veto_is_final():
    """CLAUDE.md 2: a veto cannot be talked out of."""
    arb = RiskArbiter()
    thin = Proposal("H-X", 0.05, 0.9, 50.0, MIN_TRADES_FOR_LIVE - 1, True)
    ruling = arb.rule(thin)
    assert ruling.vetoed and ruling.max_size_fraction == 0.0
    try:
        ruling.enforce()
    except RiskVeto:
        pass
    else:                                            # pragma: no cover
        raise AssertionError("enforce() let a vetoed proposal through")


def test_simulated_evidence_sizes_nothing():
    arb = RiskArbiter()
    r = arb.rule(Proposal("H-S", 0.05, 0.95, 80.0, 5000, True, is_simulated=True))
    assert r.vetoed


def test_gross_cap_is_absolute():
    arb = RiskArbiter()
    r = arb.rule(Proposal("H-G", 5.0, 1.0, 100.0, 10_000, True))
    assert r.max_size_fraction <= MAX_GROSS_FRACTION


def test_drawdown_scales_size_down():
    arb = RiskArbiter()
    flat = arb.rule(Proposal("H-D", 0.10, 0.8, 30.0, 2000, True, current_drawdown=0.0))
    deep = arb.rule(Proposal("H-D", 0.10, 0.8, 30.0, 2000, True, current_drawdown=0.15))
    assert deep.max_size_fraction < flat.max_size_fraction


def test_risk_arbiter_imports_nothing_from_stark():
    """CLAUDE.md 1: the arbiter must stay dependency-free."""
    src = (ROOT / "core" / "risk_arbiter.py").read_text()
    for pkg in ("core.", "simulation.", "tensors.", "interface."):
        assert f"import {pkg}" not in src and f"from {pkg}" not in src


def test_no_bare_global_rng():
    """CLAUDE.md 4: every stochastic call takes an explicit seed."""
    for path in ROOT.rglob("*.py"):
        if path == pathlib.Path(__file__).resolve():
            continue  # this file names the banned calls in order to ban them
        text = path.read_text()
        for bad in ("np.random.seed", "np.random.normal(", "np.random.rand("):
            assert bad not in text, f"{path.name} uses the global RNG ({bad})"


# ── level 2 ───────────────────────────────────────────────────────────────────

def _toy_graph() -> CausalGraph:
    return (
        CausalGraph()
        .add("liquidity", (), lambda v, e: 1.0 + 0.3 * e)
        .add("size", (), lambda v, e: np.full_like(e, 0.05))
        .add("impact_bps", ("size", "liquidity"),
             lambda v, e: 40.0 * v["size"] / np.maximum(v["liquidity"], 0.2))
        .add("pnl_bps", ("impact_bps",), lambda v, e: 10.0 - v["impact_bps"] + 0.2 * e)
    )


def test_do_severs_a_node_from_its_parents():
    g = _toy_graph()
    world = g.do(n=500, seed=3, size=0.2)
    assert np.allclose(world["size"], 0.2)


def test_counterfactual_reuses_this_world():
    g = _toy_graph()
    base = g.sample(n=500, seed=3)
    cf = g.counterfactual(base, size=0.0)
    assert np.allclose(base["liquidity"], cf["liquidity"])       # same world...
    assert cf.mean("pnl_bps") > base.mean("pnl_bps")             # ...different action


def test_bigger_size_costs_more_impact():
    mv = Multiverse(_toy_graph(), "pnl_bps").explore(
        {"small": {"size": 0.01}, "large": {"size": 0.20}}, n=800, seed=5
    )
    ranked = mv.ranked()
    assert ranked[0].label == "small"
    assert ranked[-1].regret_vs(mv.baseline) > 0


def test_unknown_intervention_raises():
    try:
        _toy_graph().do(n=10, seed=0, nonexistent_node=1.0)
    except ValueError:
        pass
    else:                                            # pragma: no cover
        raise AssertionError("intervened on a node that does not exist")


def test_surprise_grows_with_distance():
    loop = ActiveInferenceLoop()
    b = loop.believe("edge", 10.0, 4.0)
    assert b.surprise(10.0) < b.surprise(20.0) < b.surprise(60.0)


def test_belief_moves_toward_evidence_and_tightens():
    loop = ActiveInferenceLoop()
    loop.believe("edge", 20.0, 25.0)
    for _ in range(10):
        loop.observe(Observation("edge", 4.0, variance=9.0))
    post = loop.beliefs["edge"]
    assert 3.0 < post.mean < 8.0
    assert post.variance < 25.0


def test_curiosity_changes_what_gets_chosen():
    loop = ActiveInferenceLoop()
    loop.believe("edge", 5.0, 16.0)
    actions = [
        Action("exploit", "edge", expected_value=4.0, information_gain=0.05, cost=1.0),
        Action("probe", "edge", expected_value=0.2, information_gain=0.9, cost=0.5),
    ]
    loop.epistemic_weight = 0.1
    assert loop.select_action(actions)[0].name == "exploit"
    loop.epistemic_weight = 6.0
    assert loop.select_action(actions)[0].name == "probe"


# ── level 3 ───────────────────────────────────────────────────────────────────

def test_tensor_rejects_a_shape_mismatch():
    try:
        MarketTensor(np.zeros((2, 3, 10)), ["A"], ["p", "r", "v"])
    except ValueError:
        pass
    else:                                            # pragma: no cover
        raise AssertionError("accepted a mislabelled tensor")


def test_heat_discounts_a_stretch():
    """The whole point of F: the same displacement means less in a hot tape."""
    t = MarketTensor.from_panels({
        "CALM": synth_panel(vol=0.004, shock=0.010, seed=1),
        "HOT": synth_panel(vol=0.030, shock=0.010, seed=2),
    })
    state = {s.asset: s for s in t.field_state()}
    assert state["CALM"].disequilibrium > state["HOT"].disequilibrium


def test_free_energy_never_flips_the_sign_of_energy():
    t = MarketTensor.from_panels({
        "A": synth_panel(vol=0.004, shock=0.010, seed=1),
        "B": synth_panel(vol=0.050, shock=-0.020, seed=2),
        "C": synth_panel(vol=0.020, seed=3),
    })
    for s in t.field_state():
        assert s.free_energy == 0.0 or np.sign(s.free_energy) == np.sign(s.energy)


def test_entropy_is_normalised():
    t = MarketTensor.from_panels({"A": synth_panel(seed=1), "B": synth_panel(vol=0.05, seed=2)})
    assert np.all((t.entropy() >= 0.0) & (t.entropy() <= 1.0))


# ── level 4 ───────────────────────────────────────────────────────────────────

def test_jarvis_streams_every_layer():
    features, fwd = fixture(n=900, strength=0.2)
    j = JarvisNode(seed=1, n_permutations=300)
    out = j.run(features, fwd, tensor=MarketTensor.from_panels({"ES": synth_panel(seed=3)}))
    channels = {u.channel for u in out}
    assert {"FIELD", "IDEA", "ATTACK", "RULING", "BELIEF", "VOICE"} <= channels
    assert out[-1].channel == "VOICE"


def test_jarvis_is_deterministic():
    features, fwd = fixture(n=600)
    a = JarvisNode(seed=1, n_permutations=200).run(features, fwd)
    b = JarvisNode(seed=1, n_permutations=200).run(features, fwd)
    assert [u.text for u in a] == [u.text for u in b]


def test_jarvis_never_softens_a_veto():
    """CLAUDE.md 2 + the interface rule: JARVIS reformats, it does not contradict."""
    features, fwd = fixture(n=40, strength=0.9, seed=3)   # far too few rows for live risk
    out = JarvisNode(seed=1, n_permutations=200).run(features, fwd)
    voice = " ".join(u.text for u in out if u.channel == "VOICE").lower()
    rulings = [u for u in out if u.channel == "RULING"]
    if any(u.payload.get("decision") == "VETO" for u in rulings):
        assert "veto" in voice and "no position" in voice
    else:
        assert "risk permits" in voice or "no proposal" in voice or "died under attack" in voice


def test_jarvis_labels_simulated_numbers():
    """CLAUDE.md 3: simulated numbers say so."""
    features, fwd = fixture(n=900, strength=0.2)
    from interface.jarvis_node import _demo_graph
    out = JarvisNode(seed=1, n_permutations=300).run(features, fwd, graph=_demo_graph())
    for u in out:
        if u.channel == "BRANCH":
            assert "SIMULATED" in u.text


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failures.append((name, e))
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
