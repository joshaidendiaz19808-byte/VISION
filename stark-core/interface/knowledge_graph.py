#!/usr/bin/env python3
"""
Level 4 - the knowledge graph. JARVIS's memory, and the thing it reasons over.

A typed graph of entities and relations, persisted between sessions. Chokepoints,
commodities, refineries, macro series, vessels, hypotheses - and the relations
between them.

THE RULE THAT MAKES IT USEFUL RATHER THAN DECORATIVE

Every edge carries how it came to be believed:

    OBSERVED     a correlation measured in data. Says nothing about direction.
    DISCOVERED   an arrow oriented by structure_learning from a collider
    EXPERIMENTAL an arrow settled by an actual do() intervention
    ASSERTED     a human said so. Often right, never evidence
    SIMULATED    it happened in a model. Sizes nothing, ever
    AMBIGUOUS    adjacency is real, direction is not identified

and causal queries may ONLY traverse edges whose direction was established -
DISCOVERED, EXPERIMENTAL, or ASSERTED. An OBSERVED correlation is not a causal
edge and cannot be walked as though it were. AMBIGUOUS edges are stored, are
visible, and are refused for traversal with the experiment that would fix them.

This is the whole difference between a knowledge graph and a pile of arrows
someone drew. Walking a correlation as if it were a mechanism is how a system
confidently recommends the exact opposite of the right action.

CONFIDENCE DECAYS ALONG A PATH. A four-hop chain of 0.8-confidence edges is a
0.41-confidence claim, and the graph reports that rather than the strongest link
in it. Long causal stories feel more insightful and are worth less.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# How an edge came to be believed, and whether a causal query may walk it.
OBSERVED = "OBSERVED"
DISCOVERED = "DISCOVERED"
EXPERIMENTAL = "EXPERIMENTAL"
ASSERTED = "ASSERTED"
SIMULATED = "SIMULATED"
AMBIGUOUS = "AMBIGUOUS"

TRAVERSABLE = frozenset({DISCOVERED, EXPERIMENTAL, ASSERTED})
EVIDENCE_RANK = {EXPERIMENTAL: 5, DISCOVERED: 4, ASSERTED: 3,
                 OBSERVED: 2, SIMULATED: 1, AMBIGUOUS: 0}


@dataclass(frozen=True)
class Entity:
    """A thing the system knows about."""

    entity_id: str
    kind: str                       # chokepoint | commodity | facility | series | vessel | book
    name: str
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name} ({self.kind})"


@dataclass(frozen=True)
class Edge:
    """A believed relation, and the receipt for why it is believed."""

    source: str
    target: str
    relation: str                   # causes | supplies | correlates_with | hedges | transits
    evidence: str
    weight: float = 0.0             # signed effect size, in the units of the relation
    confidence: float = 0.5         # [0, 1]
    source_note: str = ""           # where it came from, in words
    updated: float = field(default_factory=time.time)

    @property
    def traversable(self) -> bool:
        """May a causal query walk this edge?"""
        return self.relation == "causes" and self.evidence in TRAVERSABLE

    def age_days(self, now: float | None = None) -> float:
        return ((now or time.time()) - self.updated) / 86_400.0

    def __str__(self) -> str:
        arrow = "->" if self.traversable else ("--" if self.evidence == AMBIGUOUS else "..")
        return (f"{self.source} {arrow} {self.target} [{self.relation}/{self.evidence}] "
                f"w={self.weight:+.2f} c={self.confidence:.2f}")


@dataclass(frozen=True)
class CausalPath:
    """A chain of established causal edges, with honestly compounded confidence."""

    edges: tuple[Edge, ...]

    @property
    def nodes(self) -> tuple[str, ...]:
        return (self.edges[0].source,) + tuple(e.target for e in self.edges)

    @property
    def hops(self) -> int:
        return len(self.edges)

    @property
    def confidence(self) -> float:
        """Product of the links. A chain is weaker than its weakest link, not equal to it."""
        total = 1.0
        for edge in self.edges:
            total *= edge.confidence
        return float(total)

    @property
    def effect(self) -> float:
        """Effects multiply along the chain. Signs matter: two negatives are a positive."""
        total = 1.0
        for edge in self.edges:
            total *= edge.weight
        return float(total)

    @property
    def weakest_evidence(self) -> str:
        return min((e.evidence for e in self.edges), key=lambda e: EVIDENCE_RANK[e])

    def describe(self, graph: "KnowledgeGraph") -> str:
        names = [graph.name_of(n) for n in self.nodes]
        return (" -> ".join(names) +
                f"  (effect {self.effect:+.2f}, confidence {self.confidence:.2f} "
                f"over {self.hops} hop{'s' if self.hops > 1 else ''}, "
                f"weakest link {self.weakest_evidence})")


class KnowledgeGraph:
    """
    Persistent, typed, provenance-carrying. JARVIS's world model.

    It is memory, not intelligence. It stores what was established and by what
    means, and it refuses questions the stored evidence cannot answer.
    """

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.edges: list[Edge] = []

    # ── building ─────────────────────────────────────────────────────────────
    def add_entity(self, entity_id: str, kind: str, name: str, **attributes) -> Entity:
        entity = Entity(entity_id, kind, name, dict(attributes))
        self.entities[entity_id] = entity
        return entity

    def add_edge(self, source: str, target: str, relation: str, evidence: str,
                 weight: float = 0.0, confidence: float = 0.5,
                 source_note: str = "", updated: float | None = None) -> Edge:
        for node in (source, target):
            if node not in self.entities:
                raise KeyError(f"unknown entity '{node}' - add it before linking it")
        if evidence not in EVIDENCE_RANK:
            raise ValueError(f"unknown evidence kind '{evidence}'")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence {confidence} outside [0, 1]")
        edge = Edge(source, target, relation, evidence, float(weight), float(confidence),
                    source_note, updated if updated is not None else time.time())
        self.edges.append(edge)
        return edge

    def ingest_discovered_graph(self, discovered, confidence: float = 0.7,
                                note: str = "structure_learning") -> list[Edge]:
        """
        Absorb a `DiscoveredGraph` from structure_learning.

        Oriented edges become DISCOVERED and are traversable. Undirected edges
        become AMBIGUOUS and are NOT - the adjacency is real, the direction was
        never established, and the graph keeps that distinction rather than
        flattening it into a confident arrow.
        """
        added = []
        for parent, child in sorted(discovered.directed):
            evidence = EXPERIMENTAL if "intervention" in discovered.evidence.get(
                (parent, child), "") else DISCOVERED
            added.append(self.add_edge(parent, child, "causes", evidence,
                                       weight=1.0, confidence=confidence, source_note=note))
        for pair in discovered.undirected:
            a, b = sorted(pair)
            added.append(self.add_edge(a, b, "causes", AMBIGUOUS, weight=1.0,
                                       confidence=confidence * 0.5,
                                       source_note=f"{note}: direction not identified"))
        return added

    # ── querying ─────────────────────────────────────────────────────────────
    def name_of(self, entity_id: str) -> str:
        entity = self.entities.get(entity_id)
        return entity.name if entity else entity_id

    def of_kind(self, kind: str) -> list[Entity]:
        return sorted((e for e in self.entities.values() if e.kind == kind),
                      key=lambda e: e.name)

    def edges_from(self, entity_id: str, traversable_only: bool = False) -> list[Edge]:
        return [e for e in self.edges
                if e.source == entity_id and (e.traversable or not traversable_only)]

    def ambiguous_edges(self) -> list[Edge]:
        return [e for e in self.edges if e.evidence == AMBIGUOUS]

    def causal_paths(self, source: str, target: str | None = None,
                     max_hops: int = 4, min_confidence: float = 0.02) -> list[CausalPath]:
        """
        Every established causal route out of `source`.

        Walks ONLY traversable edges. A correlation in the graph is invisible to
        this method by design - if you want it to be walkable, establish the
        direction and store it as DISCOVERED or EXPERIMENTAL.
        """
        if source not in self.entities:
            raise KeyError(f"unknown entity '{source}'")
        found: list[CausalPath] = []
        stack: list[tuple[str, tuple[Edge, ...], set[str]]] = [(source, (), {source})]
        while stack:
            node, chain, seen = stack.pop()
            for edge in self.edges_from(node, traversable_only=True):
                if edge.target in seen:
                    continue                      # no cycles, no double counting
                path = CausalPath(chain + (edge,))
                if path.confidence < min_confidence:
                    continue
                if target is None or edge.target == target:
                    found.append(path)
                if path.hops < max_hops:
                    stack.append((edge.target, path.edges, seen | {edge.target}))
        return sorted(found, key=lambda p: abs(p.effect) * p.confidence, reverse=True)

    def blocked_questions(self, source: str) -> list[str]:
        """
        What this graph cannot answer from `source`, and what would fix it.

        The most useful output here. Every ambiguous edge is a question the data
        did not settle, and each one names the experiment that would.
        """
        out = []
        for edge in self.ambiguous_edges():
            if source not in (edge.source, edge.target):
                continue
            a, b = self.name_of(edge.source), self.name_of(edge.target)
            out.append(f"'{a}' and '{b}' are related but the direction was never "
                       f"established. Intervene on '{a}' and see whether '{b}' moves.")
        return out

    def stale_edges(self, max_age_days: float = 90.0, now: float | None = None) -> list[Edge]:
        """Relations nobody has re-checked. A graph that never forgets is a graph that lies."""
        return [e for e in self.edges if e.age_days(now) > max_age_days]

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entities": [asdict(e) for e in self.entities.values()],
            "edges": [asdict(e) for e in self.edges],
            "saved_at": time.time(),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return len(self.entities) + len(self.edges)

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        graph = cls()
        payload = json.loads(Path(path).read_text())
        for row in payload.get("entities", []):
            graph.entities[row["entity_id"]] = Entity(
                row["entity_id"], row["kind"], row["name"], row.get("attributes", {}))
        for row in payload.get("edges", []):
            graph.edges.append(Edge(**row))
        return graph

    def summary(self) -> str:
        kinds: dict[str, int] = {}
        for entity in self.entities.values():
            kinds[entity.kind] = kinds.get(entity.kind, 0) + 1
        evidence: dict[str, int] = {}
        for edge in self.edges:
            evidence[edge.evidence] = evidence.get(edge.evidence, 0) + 1
        return (f"{len(self.entities)} entities ("
                + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) + ")\n"
                + f"{len(self.edges)} edges ("
                + ", ".join(f"{n} {k}" for k, n in sorted(evidence.items(),
                                                          key=lambda kv: -EVIDENCE_RANK[kv[0]]))
                + f"), {len(self.ambiguous_edges())} of them not traversable")


def demo_graph() -> KnowledgeGraph:
    """A small world model: a chokepoint, what flows through it, and what that touches."""
    graph = KnowledgeGraph()
    graph.add_entity("malacca", "chokepoint", "Strait of Malacca",
                     daily_transit_mbd=16.0, latitude=1.43, longitude=102.9)
    graph.add_entity("crude_freight", "rate", "VLCC freight rate, TD3C")
    graph.add_entity("brent", "commodity", "Brent crude")
    graph.add_entity("sg_refining", "facility", "Singapore refining complex")
    graph.add_entity("asia_diesel_crack", "commodity", "Asia diesel crack")
    graph.add_entity("jpy", "series", "USD/JPY")
    graph.add_entity("energy_book", "book", "Energy book")

    graph.add_edge("malacca", "crude_freight", "causes", EXPERIMENTAL, weight=1.8,
                   confidence=0.85, source_note="2021 Suez analogue; do() on transit capacity")
    graph.add_edge("crude_freight", "brent", "causes", DISCOVERED, weight=0.35,
                   confidence=0.70, source_note="collider orientation, 2019-2024 daily")
    graph.add_edge("malacca", "sg_refining", "causes", EXPERIMENTAL, weight=-1.2,
                   confidence=0.80, source_note="feedstock arrival model, do() on transit")
    graph.add_edge("sg_refining", "asia_diesel_crack", "causes", DISCOVERED, weight=-0.9,
                   confidence=0.75, source_note="run-cut elasticity, 2018-2024")
    graph.add_edge("brent", "energy_book", "causes", ASSERTED, weight=1.0, confidence=0.9,
                   source_note="book construction: we are long crude beta")
    graph.add_edge("asia_diesel_crack", "energy_book", "causes", ASSERTED, weight=0.6,
                   confidence=0.9, source_note="book construction: long crack spread")
    # a real correlation nobody has oriented - stored, visible, NOT traversable
    graph.add_edge("brent", "jpy", "correlates_with", OBSERVED, weight=0.31,
                   confidence=0.6, source_note="rolling 250d correlation")
    graph.add_edge("malacca", "jpy", "causes", AMBIGUOUS, weight=0.4, confidence=0.35,
                   source_note="direction not identified from observation")
    return graph


if __name__ == "__main__":
    print("STARK MK18 - knowledge graph\n")
    graph = demo_graph()
    print(graph.summary())

    print("\nCausal routes out of the Strait of Malacca:")
    for path in graph.causal_paths("malacca", target="energy_book"):
        print(f"  {path.describe(graph)}")

    print("\nNote what is NOT in that list:")
    print("  brent .. jpy is a real 0.31 correlation, stored and visible, and the")
    print("  causal query cannot walk it. Correlation is not a mechanism.")
    for question in graph.blocked_questions("malacca"):
        print(f"  ! {question}")

    print("\nAnd note the compounding: a 2-hop route at 0.85 and 0.70 is a 0.60 claim,")
    print("not a 0.85 one. Long causal stories feel more insightful and are worth less.")
