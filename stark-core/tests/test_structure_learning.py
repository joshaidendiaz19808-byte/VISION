#!/usr/bin/env python3
"""
Causal structure discovery: does it find the right graph, and does it admit
what it cannot find?

The second half matters more than the first. A discovery algorithm that always
returns a fully oriented graph is not confident, it is broken - observational
data genuinely cannot orient a chain, and the correct output there is "I don't
know, here is the experiment that would tell you".

Run: pytest stark-core/tests/test_structure_learning.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.causal_engine import CausalGraph
from simulation.structure_learning import (
    DiscoveredGraph,
    discover_structure,
    independence_p_value,
    orient_with_interventions,
    partial_correlation,
)

N = 4_000


@pytest.fixture
def collider_data():
    """liquidity -> flow <- news, flow -> price. A collider: orientable by observation."""
    rng = np.random.default_rng(7)
    liquidity, news = rng.normal(0, 1, N), rng.normal(0, 1, N)
    flow = 0.8 * liquidity + 0.9 * news + rng.normal(0, 0.4, N)
    return {"liquidity": liquidity, "news": news, "flow": flow,
            "price": 1.2 * flow + rng.normal(0, 0.5, N)}


@pytest.fixture
def chain_data():
    """supply -> inventory -> spot. No collider: NOT orientable by observation."""
    rng = np.random.default_rng(11)
    supply = rng.normal(0, 1, N)
    inventory = 0.9 * supply + rng.normal(0, 0.4, N)
    return {"supply": supply, "inventory": inventory,
            "spot": 1.1 * inventory + rng.normal(0, 0.4, N)}


def chain_truth() -> CausalGraph:
    return (CausalGraph()
            .add("supply", (), lambda v, e: e)
            .add("inventory", ("supply",), lambda v, e: 0.9 * v["supply"] + 0.4 * e)
            .add("spot", ("inventory",), lambda v, e: 1.1 * v["inventory"] + 0.4 * e))


# ── the independence test underneath ──────────────────────────────────────────

class TestIndependence:
    def test_conditioning_on_the_common_cause_kills_the_correlation(self):
        """The mechanism the whole algorithm rests on."""
        rng = np.random.default_rng(1)
        cause = rng.normal(0, 1, N)
        a, b = cause + rng.normal(0, 0.3, N), cause + rng.normal(0, 0.3, N)
        assert abs(partial_correlation(a, b)) > 0.8                    # look related...
        assert abs(partial_correlation(a, b, cause)) < 0.1             # ...but they are not

    def test_conditioning_on_a_collider_CREATES_correlation(self):
        """
        The fingerprint that makes orientation possible: two independent causes
        become dependent once you condition on their shared effect. Nothing but
        a collider does this, which is why colliders are identifiable.
        """
        rng = np.random.default_rng(2)
        a, b = rng.normal(0, 1, N), rng.normal(0, 1, N)
        collider = a + b + rng.normal(0, 0.2, N)
        assert abs(partial_correlation(a, b)) < 0.1
        assert abs(partial_correlation(a, b, collider)) > 0.5

    def test_p_value_is_large_for_independent_series(self):
        rng = np.random.default_rng(3)
        r = partial_correlation(rng.normal(0, 1, N), rng.normal(0, 1, N))
        assert independence_p_value(r, N, 0) > 0.05

    def test_p_value_is_tiny_for_a_real_relationship(self):
        rng = np.random.default_rng(4)
        x = rng.normal(0, 1, N)
        assert independence_p_value(partial_correlation(x, 0.5 * x + rng.normal(0, 1, N)), N, 0) < 1e-6

    def test_too_few_rows_refuses_to_answer(self):
        assert independence_p_value(0.9, 3, 1) == 1.0


# ── discovery from observation ────────────────────────────────────────────────

class TestDiscovery:
    def test_finds_the_true_skeleton(self, collider_data):
        found = discover_structure(collider_data)
        for a, b in (("liquidity", "flow"), ("news", "flow"), ("flow", "price")):
            assert found.has_edge(a, b), f"missed the real edge {a}-{b}"

    def test_does_not_invent_edges_between_independent_variables(self, collider_data):
        found = discover_structure(collider_data)
        assert not found.has_edge("liquidity", "news"), "invented an edge between independent causes"
        assert not found.has_edge("liquidity", "price"), "kept an edge that flow explains away"

    def test_orients_a_collider_from_observation_alone(self, collider_data):
        found = discover_structure(collider_data)
        assert ("liquidity", "flow") in found.directed
        assert ("news", "flow") in found.directed
        assert found.parents_of("flow") == ("liquidity", "news")

    def test_a_chain_is_left_undirected_because_it_HAS_to_be(self, chain_data):
        """
        The honesty test. supply->inventory->spot and spot->inventory->supply are
        Markov equivalent - identical joint distributions. Any algorithm that
        confidently orients this from observation is lying.
        """
        found = discover_structure(chain_data)
        assert found.has_edge("supply", "inventory")
        assert found.has_edge("inventory", "spot")
        assert not found.fully_oriented
        assert len(found.undirected) == 2

    def test_it_says_which_experiment_would_settle_it(self, chain_data):
        questions = discover_structure(chain_data).open_questions()
        assert len(questions) == 2
        assert all("Intervene on" in q for q in questions)

    def test_it_refuses_to_build_an_scm_from_an_undecided_graph(self, chain_data):
        found = discover_structure(chain_data)
        with pytest.raises(ValueError, match="undirected"):
            found.to_causal_graph(chain_data)

    def test_pure_noise_produces_no_edges(self):
        rng = np.random.default_rng(5)
        noise = {f"n{i}": rng.normal(0, 1, N) for i in range(4)}
        assert discover_structure(noise, alpha=0.01).n_edges == 0

    def test_alpha_trades_one_error_for_the_other(self, collider_data):
        strict = discover_structure(collider_data, alpha=1e-8)
        loose = discover_structure(collider_data, alpha=0.2)
        assert strict.n_edges <= loose.n_edges

    def test_too_little_data_refuses_rather_than_guesses(self):
        rng = np.random.default_rng(6)
        with pytest.raises(ValueError, match="guesswork"):
            discover_structure({"a": rng.normal(0, 1, 20), "b": rng.normal(0, 1, 20)})

    def test_ragged_input_is_rejected(self):
        rng = np.random.default_rng(6)
        with pytest.raises(ValueError):
            discover_structure({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 90)})

    def test_discovery_is_deterministic(self, collider_data):
        first, second = discover_structure(collider_data), discover_structure(collider_data)
        assert first.directed == second.directed and first.undirected == second.undirected


# ── interventions finish the job ──────────────────────────────────────────────

class TestInterventions:
    def _intervener(self, truth: CausalGraph):
        def intervene(variable, value, n, seed):
            return truth.do(n=n, seed=seed, **{variable: value}).values
        return intervene

    def test_experiments_orient_what_observation_could_not(self, chain_data):
        found = discover_structure(chain_data)
        assert not found.fully_oriented
        orient_with_interventions(found, self._intervener(chain_truth()), n=2_000, seed=3)
        assert found.fully_oriented
        assert ("supply", "inventory") in found.directed
        assert ("inventory", "spot") in found.directed

    def test_the_arrow_points_the_way_the_world_does(self, chain_data):
        """Forcing supply moves inventory; forcing inventory leaves supply alone."""
        found = discover_structure(chain_data)
        orient_with_interventions(found, self._intervener(chain_truth()), n=2_000, seed=3)
        assert ("inventory", "supply") not in found.directed, "arrow points backwards in time"

    def test_the_learned_graph_becomes_a_runnable_scm(self, chain_data):
        found = discover_structure(chain_data)
        orient_with_interventions(found, self._intervener(chain_truth()), n=2_000, seed=3)
        scm = found.to_causal_graph(chain_data)
        base = scm.sample(n=2_000, seed=1)
        forced = scm.do(n=2_000, seed=1, inventory=3.0)
        assert forced.mean("spot") > base.mean("spot") + 2.0
        assert forced.mean("supply") == pytest.approx(base.mean("supply"), abs=0.1), (
            "intervening downstream moved an upstream variable - the arrows are wrong"
        )

    def test_fitted_coefficients_are_close_to_the_truth(self, chain_data):
        found = discover_structure(chain_data)
        orient_with_interventions(found, self._intervener(chain_truth()), n=2_000, seed=3)
        scm = found.to_causal_graph(chain_data)
        base = scm.sample(n=20_000, seed=2)
        # truth: spot = 1.1 * inventory, inventory = 0.9 * supply, so d(spot)/d(supply) = 0.99
        lifted = scm.do(n=20_000, seed=2, supply=1.0)
        dropped = scm.do(n=20_000, seed=2, supply=0.0)
        assert (lifted.mean("spot") - dropped.mean("spot")) == pytest.approx(0.99, rel=0.15)

    def test_an_ambiguous_edge_stays_ambiguous(self):
        """
        If forcing either variable moves the other - a hidden common cause or a
        feedback loop - the honest answer is still "undecided", not a coin flip.
        """
        graph = DiscoveredGraph(variables=["a", "b"], undirected={frozenset(("a", "b"))})
        rng = np.random.default_rng(1)

        def both_ways(variable, value, n, seed):
            other = "b" if variable == "a" else "a"
            return {variable: np.full(n, value), other: value + rng.normal(0, 1, n)}

        orient_with_interventions(graph, both_ways, n=500, seed=1)
        assert not graph.fully_oriented, "picked a direction when the evidence pointed both ways"
