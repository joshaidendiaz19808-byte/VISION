#!/usr/bin/env python3
"""
Phase 3: capital fields at equilibrium, and tensor network contraction.

Two things are worth proving here and they are both proofs, not vibes:

  1. the equilibrium solver lands on the analytic Boltzmann answer in the case
     where an analytic answer exists, and never crosses a hard cap in any case
  2. contraction returns EXACTLY what brute-force enumeration returns - every
     marginal, every conditional, every joint probability

Anything calling itself a tensor network that cannot match enumeration on a
small case is not computing what it says it is computing.

Run: pytest stark-core/tests/test_tensors.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from tensors.market_tensors import (
    CapitalField,
    MarketStateNetwork,
    MarketTensor,
    StateVariable,
    project_to_capped_simplex,
    synth_panel,
)


# ── the projection: where hard limits are enforced ────────────────────────────

class TestProjection:
    def test_result_is_always_feasible(self):
        rng = np.random.default_rng(1)
        caps = np.array([0.4, 0.3, 0.5, 0.25])
        for _ in range(200):
            w = project_to_capped_simplex(rng.normal(0, 2, 4), caps)
            assert w.sum() == pytest.approx(1.0, abs=1e-9)
            assert np.all(w >= -1e-12)
            assert np.all(w <= caps + 1e-12), "projection produced a weight above its cap"

    def test_an_already_feasible_point_is_left_alone(self):
        w = np.array([0.2, 0.3, 0.5])
        assert np.allclose(project_to_capped_simplex(w, np.ones(3)), w, atol=1e-9)

    def test_it_finds_the_NEAREST_feasible_point(self):
        """Projection means nearest. Verified against a dense search."""
        target = np.array([0.9, 0.05, 0.05])
        caps = np.array([0.5, 0.5, 0.5])
        got = project_to_capped_simplex(target, caps)
        rng = np.random.default_rng(2)
        for _ in range(600):
            other = project_to_capped_simplex(rng.dirichlet(np.ones(3)) + rng.normal(0, .3, 3), caps)
            assert np.linalg.norm(got - target) <= np.linalg.norm(other - target) + 1e-9

    def test_infeasible_limits_are_refused_not_fudged(self):
        with pytest.raises(ValueError, match="infeasible"):
            project_to_capped_simplex(np.ones(3), np.array([0.1, 0.1, 0.1]))


# ── the capital field ─────────────────────────────────────────────────────────

def field(**kw) -> CapitalField:
    base = dict(names=["a", "b", "c", "d"], hypotheses=[20.0, 12.0, 8.0, 4.0],
                caps=1.0, risk_aversion=0.0)
    base.update(kw)
    return CapitalField(**base)


class TestEquilibrium:
    def test_it_matches_the_analytic_boltzmann_solution(self):
        """
        With no covariance term and no binding cap, minimising F = U - T*S has a
        closed form: w ∝ exp(edge / T). If the solver does not land there, it is
        not minimising what it claims to minimise.
        """
        f = field()
        for temperature in (20.0, 10.0, 5.0):
            solved = f.equilibrium(temperature=temperature, max_iterations=40_000, step=0.01)
            assert solved.converged
            assert np.allclose(solved.weights, f.boltzmann(temperature), atol=0.02), (
                f"T={temperature}: {solved.weights} vs analytic {f.boltzmann(temperature)}"
            )

    def test_the_solution_is_a_global_minimum(self):
        """
        F is convex on a convex feasible set, so the solver's answer must beat
        every other feasible point - not merely be a place the gradient stopped.
        """
        f = field(risk_aversion=1.0)
        solved = f.equilibrium(temperature=5.0)
        best = f.free_energy(solved.weights, 5.0)
        rng = np.random.default_rng(4)
        for _ in range(2_000):
            other = project_to_capped_simplex(rng.dirichlet(np.ones(4)), np.ones(4))
            assert best <= f.free_energy(other, 5.0) + 1e-6

    def test_descent_is_monotone_at_a_stable_step(self):
        """
        Plain projected gradient descent decreases F when the step is small
        enough. (The solver does not rely on this - it backtracks whenever a
        step would raise F - but the gradient itself must point downhill.)
        """
        f = field(risk_aversion=1.0)
        w = project_to_capped_simplex(np.array([0.7, 0.1, 0.1, 0.1]), np.ones(4))
        history = [f.free_energy(w, 5.0)]
        for _ in range(60):
            w = project_to_capped_simplex(w - 0.001 * f.gradient(w, 5.0), np.ones(4))
            history.append(f.free_energy(w, 5.0))
        assert all(b <= a + 1e-9 for a, b in zip(history, history[1:])), "F went uphill"

    def test_capital_is_conserved(self):
        for temperature in (0.5, 5.0, 50.0):
            assert field().equilibrium(temperature=temperature).weights.sum() == pytest.approx(1.0)

    def test_caps_are_never_crossed_at_any_temperature(self):
        """Rule 2, restated as thermodynamics: no temperature buys extra size."""
        caps = np.array([0.30, 0.30, 0.30, 0.10])
        f = field(caps=caps)
        for temperature in (0.01, 0.5, 2.0, 20.0, 500.0):
            w = f.equilibrium(temperature=temperature).weights
            assert np.all(w <= caps + 1e-9), f"T={temperature} broke a cap: {w}"

    def test_the_best_idea_is_capped_like_any_other(self):
        f = field(caps=np.array([0.10, 1.0, 1.0, 1.0]))     # cap the STRONGEST hypothesis
        solved = f.equilibrium(temperature=0.5)
        assert solved.weights[0] == pytest.approx(0.10, abs=1e-6)
        assert "a" in solved.binding

    def test_heat_diversifies_and_cold_concentrates(self):
        f = field()
        hot = f.equilibrium(temperature=100.0)
        cold = f.equilibrium(temperature=1.0)
        assert hot.entropy > cold.entropy
        assert hot.weights.max() < cold.weights.max()

    def test_at_infinite_temperature_everything_is_equal_weight(self):
        w = field().equilibrium(temperature=1e6).weights
        assert np.allclose(w, 0.25, atol=1e-3)

    def test_at_zero_temperature_capital_goes_to_the_best_idea(self):
        w = field().equilibrium(temperature=1e-6, max_iterations=20_000).weights
        assert np.argmax(w) == 0 and w[0] > 0.95

    def test_the_answer_does_not_depend_on_where_you_start(self):
        """F is convex on a convex set, so the minimum is unique. Verify it."""
        f = field(risk_aversion=1.0)
        rng = np.random.default_rng(3)
        reference = f.equilibrium(temperature=5.0).weights
        for _ in range(6):
            start = rng.dirichlet(np.ones(4))
            assert np.allclose(f.equilibrium(temperature=5.0, start=start).weights,
                               reference, atol=1e-3)

    def test_risk_aversion_pushes_capital_apart(self):
        cov = np.array([[1.0, 0.9, 0.0, 0.0], [0.9, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        timid = CapitalField(["a", "b", "c", "d"], [20.0, 19.0, 8.0, 4.0],
                             covariance=cov, risk_aversion=40.0).equilibrium(temperature=2.0)
        bold = CapitalField(["a", "b", "c", "d"], [20.0, 19.0, 8.0, 4.0],
                            covariance=cov, risk_aversion=0.0).equilibrium(temperature=2.0)
        assert timid.weights[:2].sum() < bold.weights[:2].sum(), (
            "correlated ideas were not penalised for being the same bet"
        )

    def test_mismatched_field_shapes_are_rejected(self):
        with pytest.raises(ValueError):
            CapitalField(["a", "b"], [1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            CapitalField(["a", "b"], [1.0, 2.0], caps=[0.5])


# ── tensor network contraction ────────────────────────────────────────────────

def small_network() -> MarketStateNetwork:
    return MarketStateNetwork(
        [StateVariable.from_probabilities("liquidity", {"thin": 0.3, "normal": 0.5, "deep": 0.2}),
         StateVariable.from_probabilities("regime", {"calm": 0.6, "stressed": 0.4}),
         StateVariable.from_probabilities("direction", {"down": 0.35, "flat": 0.3, "up": 0.35})],
        [np.array([[0.5, 2.0], [1.0, 1.0], [2.0, 0.5]]),
         np.array([[1.0, 1.2, 1.0], [1.6, 0.8, 0.6]])])


class TestContraction:
    def test_marginals_match_brute_force_enumeration_exactly(self):
        """The test that makes the whole module trustworthy."""
        net = small_network()
        configs, probs = net.enumerate_exactly()
        for variable in net.variables:
            contracted = np.array(list(net.marginal(variable.name).values()))
            enumerated = np.array([sum(p for c, p in zip(configs, probs) if c[variable.name] == s)
                                   for s in variable.states])
            assert np.allclose(contracted, enumerated, atol=1e-12)

    def test_joint_probabilities_match_enumeration(self):
        net = small_network()
        configs, probs = net.enumerate_exactly()
        for config, p in zip(configs, probs):
            assert net.probability(config) == pytest.approx(p, abs=1e-12)

    def test_the_cloud_sums_to_one(self):
        _, probs = small_network().enumerate_exactly()
        assert probs.sum() == pytest.approx(1.0)

    def test_conditioning_matches_enumeration(self):
        net = small_network()
        configs, probs = net.enumerate_exactly()
        conditioned = net.condition(liquidity="thin")
        mass = sum(p for c, p in zip(configs, probs) if c["liquidity"] == "thin")
        for state in ("calm", "stressed"):
            expected = sum(p for c, p in zip(configs, probs)
                           if c["liquidity"] == "thin" and c["regime"] == state) / mass
            assert conditioned.marginal("regime")[state] == pytest.approx(expected, abs=1e-12)

    def test_conditioning_propagates_through_the_coupling(self):
        net = small_network()
        assert net.condition(liquidity="thin").marginal("regime")["stressed"] > \
               net.marginal("regime")["stressed"]
        assert net.condition(liquidity="deep").marginal("regime")["stressed"] < \
               net.marginal("regime")["stressed"]

    def test_conditioning_leaves_the_original_untouched(self):
        net = small_network()
        before = net.marginal("regime")
        net.condition(liquidity="thin")
        assert net.marginal("regime") == before

    def test_no_coupling_means_variables_stay_independent(self):
        net = MarketStateNetwork(
            [StateVariable.from_probabilities("a", {"x": 0.3, "y": 0.7}),
             StateVariable.from_probabilities("b", {"x": 0.4, "y": 0.6})])
        assert net.marginal("a")["x"] == pytest.approx(0.3)
        assert net.condition(b="x").marginal("a")["x"] == pytest.approx(0.3), (
            "an uncoupled variable moved when its neighbour was observed"
        )

    def test_the_most_likely_state_is_actually_the_most_likely(self):
        net = small_network()
        configs, probs = net.enumerate_exactly()
        best_config, best_p = net.most_likely_states(1)[0]
        assert best_p == pytest.approx(probs.max(), abs=1e-12)
        assert best_config == configs[int(np.argmax(probs))]

    def test_top_k_is_the_true_top_k_in_order(self):
        net = small_network()
        _, probs = net.enumerate_exactly()
        top = [p for _, p in net.most_likely_states(5)]
        assert np.allclose(top, np.sort(probs)[::-1][:5], atol=1e-12)

    def test_samples_converge_on_the_exact_marginals(self):
        net = small_network()
        draws = net.sample(n=20_000, seed=4)
        for state, exact in net.marginal("regime").items():
            empirical = sum(d["regime"] == state for d in draws) / len(draws)
            assert empirical == pytest.approx(exact, abs=0.02)

    def test_sampling_is_deterministic(self):
        net = small_network()
        assert net.sample(50, seed=9) == net.sample(50, seed=9)
        assert net.sample(50, seed=9) != net.sample(50, seed=10)

    def test_expected_value_uses_the_contracted_marginal(self):
        net = small_network()
        payoffs = {"down": -100.0, "flat": 0.0, "up": 100.0}
        marginal = net.marginal("direction")
        assert net.expected_value("direction", payoffs) == pytest.approx(
            sum(marginal[s] * payoffs[s] for s in marginal))

    def test_a_missing_payoff_is_an_error_not_a_zero(self):
        with pytest.raises(ValueError):
            small_network().expected_value("direction", {"down": -1.0, "up": 1.0})

    def test_correlation_length_rises_with_coupling_strength(self):
        def chain(strength):
            return MarketStateNetwork(
                [StateVariable.from_probabilities(f"v{i}", {"lo": 0.5, "hi": 0.5}) for i in range(10)],
                [np.array([[strength, 1.0], [1.0, strength]])] * 9)
        assert chain(1.2).correlation_length() < chain(3.0).correlation_length()

    def test_correlation_length_is_none_for_an_inhomogeneous_chain(self):
        net = small_network()
        assert net.correlation_length() is None, (
            "reported a correlation length for a chain that has no single transfer matrix"
        )

    def test_a_large_network_is_tractable(self):
        """40 binary variables: a trillion configurations, contracted in milliseconds."""
        big = MarketStateNetwork(
            [StateVariable.from_probabilities(f"v{i}", {"lo": 0.45, "hi": 0.55}) for i in range(40)],
            [np.array([[1.6, 0.6], [0.6, 1.6]])] * 39)
        assert big.n_configurations == 2 ** 40
        marginal = big.marginal("v20")
        assert sum(marginal.values()) == pytest.approx(1.0)
        assert big.most_likely_states(1)[0][1] > 0

    def test_enumeration_refuses_to_run_on_a_large_network(self):
        big = MarketStateNetwork(
            [StateVariable.from_probabilities(f"v{i}", {"lo": 0.5, "hi": 0.5}) for i in range(30)])
        with pytest.raises(ValueError, match="too many"):
            big.enumerate_exactly()

    def test_malformed_networks_are_rejected(self):
        v = [StateVariable.from_probabilities("a", {"x": 0.5, "y": 0.5}),
             StateVariable.from_probabilities("b", {"x": 0.5, "y": 0.5})]
        with pytest.raises(ValueError, match="coupling"):
            MarketStateNetwork(v, [np.ones((3, 2))])
        with pytest.raises(ValueError, match="negative"):
            MarketStateNetwork(v, [np.array([[1.0, -1.0], [1.0, 1.0]])])
        with pytest.raises(KeyError):
            MarketStateNetwork(v).marginal("nope")
        with pytest.raises(ValueError):
            MarketStateNetwork(v).condition(a="not_a_state")

    def test_the_substrate_stays_numpy_only(self):
        """
        CLAUDE.md rule 1: tensors/ is the floor of the building. Phase 3 added a
        lot of machinery to this module and none of it was allowed to bring a
        dependency - scipy has logsumexp and eigenvalue routines, and this file
        writes its own rather than reach for them.
        """
        import ast
        import pathlib as _pl

        tree = ast.parse((_pl.Path(__file__).resolve().parents[1]
                          / "tensors" / "market_tensors.py").read_text())
        allowed = {"__future__", "dataclasses", "typing", "numpy", "itertools", "time"}
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported <= allowed, f"tensors/ picked up {imported - allowed}"

    def test_phase_one_thermodynamic_fields_still_work(self):
        """Phase 3 added to this module; it must not have broken what was here."""
        t = MarketTensor.from_panels({"A": synth_panel(vol=0.004, shock=0.010, seed=1),
                                      "B": synth_panel(vol=0.030, seed=2)})
        assert len(t.field_state()) == 2
        assert np.all((t.entropy() >= 0) & (t.entropy() <= 1))
