#!/usr/bin/env python3
"""
Phase 5 tests: the knowledge graph, overnight runs, and the JARVIS interface.

The interesting tests here are the REFUSALS. An interface that always produces a
confident answer is worse than useless, because the confidence is uncorrelated
with whether it knows anything. Most of what follows checks that it declines when
it should, and says why.

Run: pytest stark-core/tests/test_interface.py -q
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from interface.jarvis_node import (Adjustment, ApprovalRequired, JarvisInterface,
                                   JarvisNode, SessionState, _tokens)
from interface.knowledge_graph import (AMBIGUOUS, ASSERTED, DISCOVERED, EXPERIMENTAL,
                                       OBSERVED, KnowledgeGraph, demo_graph)
from interface.overnight import (DEFAULT_SCENARIOS, OvernightRunner, OvernightStore,
                                 Scenario, book_state_hash, demo_model)
from simulation.structure_learning import discover_structure


@pytest.fixture
def graph() -> KnowledgeGraph:
    return demo_graph()


@pytest.fixture
def positions() -> dict:
    return {"crude": 0.06, "diesel_crack": 0.03}


@pytest.fixture
def store(positions) -> OvernightStore:
    return OvernightRunner(demo_model(), samples=800, seed=11).run_all(
        DEFAULT_SCENARIOS, positions)


def jarvis(graph, store, positions, **kw) -> JarvisInterface:
    session = SessionState(operator="tester", positions=positions, drawdown=0.0)
    return JarvisInterface(graph, store, session, **kw)


# ── the knowledge graph ───────────────────────────────────────────────────────

class TestKnowledgeGraph:
    def test_causal_queries_refuse_to_walk_a_correlation(self, graph):
        """
        The rule the whole graph exists to enforce. brent--jpy is a real measured
        correlation; walking it as a mechanism is how a system confidently
        recommends the opposite of the right thing.
        """
        assert any(e.relation == "correlates_with" for e in graph.edges)
        for path in graph.causal_paths("brent"):
            assert all(edge.traversable for edge in path.edges)
            assert "jpy" not in path.nodes

    def test_an_ambiguous_edge_is_stored_but_not_traversable(self, graph):
        assert graph.ambiguous_edges()
        for path in graph.causal_paths("malacca"):
            assert AMBIGUOUS not in [e.evidence for e in path.edges]

    def test_it_names_the_experiment_that_would_unblock_it(self, graph):
        questions = graph.blocked_questions("malacca")
        assert questions and all("Intervene on" in q for q in questions)

    def test_confidence_compounds_along_a_path(self, graph):
        for path in graph.causal_paths("malacca", target="energy_book"):
            assert path.confidence <= min(e.confidence for e in path.edges) + 1e-12
            assert path.confidence == pytest.approx(
                float(np.prod([e.confidence for e in path.edges])))

    def test_a_longer_story_is_worth_less(self, graph):
        paths = graph.causal_paths("malacca", target="energy_book")
        long_path = max(paths, key=lambda p: p.hops)
        assert long_path.confidence < min(e.confidence for e in long_path.edges) or \
            long_path.hops == 1

    def test_effects_multiply_with_signs(self):
        g = KnowledgeGraph()
        for name in ("a", "b", "c"):
            g.add_entity(name, "series", name.upper())
        g.add_edge("a", "b", "causes", EXPERIMENTAL, weight=-2.0, confidence=1.0)
        g.add_edge("b", "c", "causes", EXPERIMENTAL, weight=-3.0, confidence=1.0)
        assert g.causal_paths("a", "c")[0].effect == pytest.approx(6.0)

    def test_cycles_are_not_walked_twice(self):
        g = KnowledgeGraph()
        for name in ("a", "b"):
            g.add_entity(name, "series", name)
        g.add_edge("a", "b", "causes", EXPERIMENTAL, weight=1.0, confidence=0.9)
        g.add_edge("b", "a", "causes", EXPERIMENTAL, weight=1.0, confidence=0.9)
        assert all(p.hops <= 2 for p in g.causal_paths("a"))

    def test_unknown_entities_are_refused(self, graph):
        with pytest.raises(KeyError):
            graph.causal_paths("atlantis")
        with pytest.raises(KeyError):
            graph.add_edge("atlantis", "brent", "causes", ASSERTED)

    def test_bad_evidence_or_confidence_is_refused(self, graph):
        with pytest.raises(ValueError):
            graph.add_edge("brent", "jpy", "causes", "VIBES")
        with pytest.raises(ValueError):
            graph.add_edge("brent", "jpy", "causes", ASSERTED, confidence=1.5)

    def test_discovered_structure_keeps_its_ambiguity(self):
        """Undirected edges from structure learning must not become confident arrows."""
        rng = np.random.default_rng(11)
        n = 3_000
        supply = rng.normal(0, 1, n)
        inventory = 0.9 * supply + rng.normal(0, 0.4, n)
        data = {"supply": supply, "inventory": inventory,
                "spot": 1.1 * inventory + rng.normal(0, 0.4, n)}
        discovered = discover_structure(data)
        assert not discovered.fully_oriented

        g = KnowledgeGraph()
        for name in data:
            g.add_entity(name, "series", name)
        g.ingest_discovered_graph(discovered)
        assert len(g.ambiguous_edges()) == len(discovered.undirected)
        assert g.causal_paths("supply") == [], "an unoriented edge became walkable"

    def test_stale_edges_are_reported(self, graph):
        old = time.time() - 200 * 86_400
        graph.add_edge("brent", "sg_refining", "causes", ASSERTED, confidence=0.5, updated=old)
        assert graph.stale_edges(max_age_days=90.0)

    def test_round_trip_through_disk(self, graph, tmp_path):
        path = tmp_path / "graph.json"
        graph.save(path)
        loaded = KnowledgeGraph.load(path)
        assert len(loaded.entities) == len(graph.entities)
        assert len(loaded.edges) == len(graph.edges)
        assert [str(p) for p in loaded.causal_paths("malacca")] == \
               [str(p) for p in graph.causal_paths("malacca")]


# ── overnight runs ────────────────────────────────────────────────────────────

class TestOvernight:
    def test_results_are_always_marked_simulated(self, store):
        assert store.results and all(r.simulated for r in store.results)

    def test_branches_vary_our_response_not_the_shock(self, store):
        """A weather forecast is not a decision. Every branch is an action we could take."""
        labels = {b.label for b in store.results[0].branches}
        assert labels == {"shock, flatten", "shock, halve", "shock, hold", "shock, add"}

    def test_a_result_knows_which_book_it_assumed(self, store, positions):
        result = store.results[0]
        assert result.stale_against(book_state_hash(positions)) is None
        assert "different book state" in result.stale_against(book_state_hash({"crude": 0.9}))

    def test_an_old_result_goes_stale_on_time(self, store, positions):
        result = store.results[0]
        aged = type(result)(**{**result.to_dict(),
                               "branches": result.branches,
                               "computed_at": time.time() - 100 * 3600})
        assert "hours ago" in aged.stale_against(book_state_hash(positions))

    def test_the_store_returns_nothing_rather_than_something_near(self, store):
        assert store.find("a-scenario-nobody-ran") is None
        assert store.for_entity("atlantis") == []

    def test_runs_are_deterministic(self, positions):
        one = OvernightRunner(demo_model(), samples=500, seed=3).run_one(
            DEFAULT_SCENARIOS[0], positions)
        two = OvernightRunner(demo_model(), samples=500, seed=3).run_one(
            DEFAULT_SCENARIOS[0], positions)
        assert [b.outcome_mean for b in one.branches] == [b.outcome_mean for b in two.branches]

    def test_percentiles_bracket_the_mean(self, store):
        for result in store.results:
            for branch in result.branches:
                assert branch.outcome_p05 <= branch.outcome_mean <= branch.outcome_p95

    def test_store_survives_a_round_trip(self, store, tmp_path):
        path = tmp_path / "overnight.json"
        store.save(path)
        loaded = OvernightStore.load(path)
        assert len(loaded.results) == len(store.results)
        assert loaded.results[0].branches[0].outcome_mean == \
               store.results[0].branches[0].outcome_mean


# ── the interface: what it answers ────────────────────────────────────────────

class TestJarvisAnswers:
    def test_a_full_query_streams_every_layer_and_ends_with_one_adjustment(
            self, graph, store, positions):
        console = jarvis(graph, store, positions)
        stream = list(console.ask("What is our exposure to a Strait of Malacca disruption?"))
        channels = [u.channel for u in stream]
        assert {"SCENARIO", "PATH", "COUNTERFACTUAL", "RULING", "ADJUSTMENT",
                "APPROVAL", "VOICE"} <= set(channels)
        assert channels[-1] == "VOICE"
        assert console.pending is not None

    def test_exactly_one_recommendation_not_a_menu(self, graph, store, positions):
        console = jarvis(graph, store, positions)
        stream = list(console.ask("Strait of Malacca disruption?"))
        assert sum(u.channel == "ADJUSTMENT" for u in stream) == 1

    def test_every_model_number_is_labelled_simulated(self, graph, store, positions):
        console = jarvis(graph, store, positions)
        for utterance in console.ask("Strait of Malacca disruption?"):
            if utterance.channel == "COUNTERFACTUAL" and "SKIPPED" not in utterance.text:
                assert "SIMULATED" in utterance.text

    def test_the_adjustment_says_what_would_make_it_wrong(self, graph, store, positions):
        console = jarvis(graph, store, positions)
        list(console.ask("Strait of Malacca disruption?"))
        assert len(console.pending.falsifiers) >= 3
        assert "WHAT WOULD MAKE THIS WRONG" in console.pending.card()

    def test_the_card_renders_as_a_rectangle(self, graph, store, positions):
        """A wrapped falsifier is read; a truncated one is not."""
        console = jarvis(graph, store, positions)
        list(console.ask("Strait of Malacca disruption?"))
        lines = console.pending.card().splitlines()
        assert len({len(line) for line in lines}) == 1, "the card is not a rectangle"

    def test_the_dashboard_reports_what_is_not_established(self, graph, store, positions):
        assert "NOT ESTABLISHED" in jarvis(graph, store, positions).dashboard()

    def test_tokeniser_matches_both_sides_identically(self):
        """Regression: 'USD/JPY' in a query did not match the entity named 'USD/JPY'."""
        assert _tokens("What does USD/JPY do to us?") & _tokens("USD/JPY")

    def test_it_is_deterministic(self, graph, store, positions):
        first = [u.text for u in jarvis(graph, store, positions).ask("Malacca disruption?")]
        second = [u.text for u in jarvis(graph, store, positions).ask("Malacca disruption?")]
        assert first == second


# ── the interface: what it refuses ────────────────────────────────────────────

class TestJarvisRefusals:
    def test_it_refuses_an_entity_it_does_not_know(self, graph, store, positions):
        voice = " ".join(jarvis(graph, store, positions)
                         .voice_only("What about a solar flare hitting the exchange?"))
        assert "do not have an entity" in voice
        assert jarvis(graph, store, positions).pending is None

    def test_it_refuses_to_guess_a_direction_it_never_established(
            self, graph, store, positions):
        console = jarvis(graph, store, positions)
        stream = list(console.ask("What does USD/JPY do to us?"))
        voice = " ".join(u.text for u in stream if u.channel == "VOICE")
        assert "no established causal route" in voice
        assert "will not guess" in voice
        assert console.pending is None, "recommended something down an unoriented edge"

    def test_it_refuses_when_nothing_was_modelled_overnight(self, graph, positions):
        console = jarvis(graph, OvernightStore(), positions)
        voice = " ".join(console.voice_only("Strait of Malacca disruption?"))
        assert "have not run the counterfactual" in voice
        assert "not going to improvise" in voice
        assert console.pending is None

    def test_it_refuses_a_counterfactual_computed_for_another_book(self, graph, store):
        console = jarvis(graph, store, {"crude": 0.99, "something_else": 0.4})
        stream = list(console.ask("Strait of Malacca disruption?"))
        assert any("SKIPPED" in u.text for u in stream)
        assert console.pending is None

    def test_a_veto_produces_no_recommendation_not_a_softened_one(self, graph, store,
                                                                  positions):
        """The drawdown halt is binding. JARVIS reports it and stops."""
        console = jarvis(graph, store, positions)
        console.session.drawdown = 0.25          # past the arbiter's hard halt
        stream = list(console.ask("Strait of Malacca disruption?"))
        voice = " ".join(u.text for u in stream if u.channel == "VOICE")
        assert console.pending is None
        assert "No position" in voice and "not going to argue with the arbiter" in voice


# ── the interface: the approval gate ──────────────────────────────────────────

class TestApprovalGate:
    def _pending(self, graph, store, positions, tmp_path=None):
        console = jarvis(graph, store, positions,
                         session_path=(tmp_path / "s.json") if tmp_path else None)
        list(console.ask("Strait of Malacca disruption?"))
        assert console.pending is not None
        return console

    def test_nothing_executes_without_approval(self, graph, store, positions):
        console = self._pending(graph, store, positions)
        with pytest.raises(ApprovalRequired):
            console.execute(console.pending)

    def test_a_rejected_adjustment_never_becomes_an_intent(self, graph, store, positions):
        console = self._pending(graph, store, positions)
        rejected = console.reject("tester", "not convinced")
        with pytest.raises(ApprovalRequired, match="only an APPROVED"):
            console.execute(rejected)

    def test_an_approved_adjustment_becomes_an_intent_not_an_order(self, graph, store,
                                                                   positions):
        console = self._pending(graph, store, positions)
        intent = console.execute(console.approve("tester", "fine"))
        assert intent.size_fraction > 0
        assert not hasattr(intent, "send") and not hasattr(intent, "broker")

    def test_approval_records_who_and_why(self, graph, store, positions):
        console = self._pending(graph, store, positions)
        console.approve("josh", "checked the freight leg")
        decision = console.session.decisions[-1]
        assert decision["by"] == "josh" and decision["note"] == "checked the freight leg"

    def test_approving_nothing_is_an_error(self, graph, store, positions):
        console = jarvis(graph, store, positions)
        with pytest.raises(ApprovalRequired):
            console.approve("tester")

    def test_rejections_are_remembered_and_surfaced_next_time(self, graph, store, positions):
        console = self._pending(graph, store, positions)
        console.reject("josh", "freight leg is one analogue")
        stream = list(console.ask("Strait of Malacca disruption?"))
        context = [u.text for u in stream if u.channel == "SCENARIO" and "rejected" in u.text]
        assert context and "one analogue" in context[0]

    def test_the_session_survives_a_restart(self, graph, store, positions, tmp_path):
        console = self._pending(graph, store, positions, tmp_path)
        console.reject("josh", "no")
        reloaded = SessionState.load(tmp_path / "s.json")
        assert len(reloaded.rejections()) == 1
        assert reloaded.rejections()[0]["by"] == "josh"


# ── the Phase 1-3 pipeline narrator still works ───────────────────────────────

def test_the_original_narrator_is_unbroken():
    """Phase 5 extended this module. It must not have broken what was here."""
    rng = np.random.default_rng(7)
    n = 600
    real = rng.normal(0, 1, n)
    features = {"real_signal": real, "noise": rng.normal(0, 1, n)}
    fwd = (0.2 * real + rng.normal(0, 1, n)) * 0.01
    out = JarvisNode(seed=1, n_permutations=200).run(features, fwd)
    assert out and out[-1].channel == "VOICE"
