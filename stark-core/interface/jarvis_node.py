#!/usr/bin/env python3
"""
Level 4 - Voice / spatial causal streaming interface.

JARVIS is the only part of STARK a human touches. Its job is not to decide
anything - every decision was already made below it - but to *narrate* the
decision as it happens, in the order it happens, in sentences a person can
argue with.

It emits a stream of Utterances rather than returning a report at the end.
That is deliberate: you should be able to interrupt STARK at the moment it says
"the red team just killed your best idea" instead of reading about it later.

  channel        what it is                for
  ---------------------------------------------------------------------
  FIELD          tensor readings           the spatial layer (where to look)
  IDEA           hypotheses                the neural layer
  ATTACK         red team findings         the adversarial layer
  RULING         risk arbiter decisions    the hard layer - never editorialised
  BRANCH         causal simulations        the counterfactual layer
  BELIEF         active inference updates  the learning layer
  SCENARIO       a macro question          Phase 5: what was asked
  PATH           knowledge graph routes    Phase 5: how the shock reaches us
  COUNTERFACTUAL overnight run results     Phase 5: what we precomputed
  ADJUSTMENT     the single recommendation Phase 5: what to do about it
  APPROVAL       the human decision        Phase 5: sovereign, and required
  VOICE          plain-English summary     what actually gets spoken

Rule: JARVIS may reformat anything and contradict nothing. If it ever finds
itself softening a VETO, that is a bug in this file, not a feature.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):  # run as a script: put the stark-core/ root on the path
    import pathlib as _pl, sys as _sys
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

from core.generator import Generator, Hypothesis
from core.red_team import RedTeam, RedTeamReport
from core.risk_arbiter import Proposal, RiskArbiter, RiskVeto, Ruling
from simulation.active_inference import Action, ActiveInferenceLoop, Observation
from simulation.causal_engine import CausalGraph, Multiverse
from tensors.market_tensors import MarketTensor
from interface.knowledge_graph import KnowledgeGraph, demo_graph
from interface.overnight import (DEFAULT_SCENARIOS, OvernightRunner, OvernightStore,
                                 book_state_hash, demo_model)

CHANNELS = ("FIELD", "IDEA", "ATTACK", "RULING", "BRANCH", "BELIEF",
            "SCENARIO", "PATH", "COUNTERFACTUAL", "ADJUSTMENT", "APPROVAL", "VOICE")

# How solid is a number? Every figure JARVIS speaks carries one of these.
REAL = "REAL"            # measured, point-in-time, provenance on file
EST = "est."             # derived from real data, with an error bar
SIMULATED = "SIMULATED"  # produced by a model. Sizes nothing, by rule.


@dataclass(frozen=True)
class Intent:
    """
    What an approved adjustment becomes. NOT an order.

    Defined here rather than imported from execution/ so this tree keeps its one
    dependency direction: the interface narrates and recommends, and knows
    nothing about how anything is executed.
    """

    action: str
    size_fraction: float
    reason: str
    entity: str = ""


@dataclass(frozen=True)
class Utterance:
    """One thing STARK says, tagged with where it came from."""

    channel: str
    level: int
    text: str
    payload: dict = field(default_factory=dict)

    def spoken(self) -> str:
        """Voice-only view: the sentence, no telemetry."""
        return self.text

    def __str__(self) -> str:
        return f"[L{self.level} {self.channel:<6}] {self.text}"


class JarvisNode:
    """
    Runs the MK18 loop and narrates it.

    Deterministic given `seed`. Holds no capital, places no orders, and cannot
    override a ruling - it only reports what the arbiter decided.
    """

    def __init__(
        self,
        seed: int = 0,
        cost_bps: float = 5.0,
        max_ideas: int = 4,
        n_permutations: int = 1000,
    ) -> None:
        self.seed = seed
        self.cost_bps = cost_bps
        self.generator = Generator(seed=seed, max_hypotheses=max_ideas, min_prior=0.0)
        self.red_team = RedTeam(cost_bps=cost_bps, seed=seed, n_permutations=n_permutations)
        self.arbiter = RiskArbiter()
        self.inference = ActiveInferenceLoop()
        self.transcript: list[Utterance] = []

    # ── the loop ─────────────────────────────────────────────────────────────
    def stream(
        self,
        features: Mapping[str, Sequence[float]],
        forward_returns: Sequence[float],
        tensor: MarketTensor | None = None,
        graph: CausalGraph | None = None,
        book_drawdown: float = 0.0,
        book_volatility_bps: float = 0.0,
        book_weight: float = 0.0,
        requested_size: float = 0.05,
    ) -> Iterator[Utterance]:
        """Yield the whole reasoning chain, in order, as it is produced."""

        # Level 3 - where should we even be looking?
        if tensor is not None:
            yield self._say("FIELD", 3, "Reading the field.")
            for state in tensor.disequilibrium_ranking():
                yield self._say(
                    "FIELD", 3,
                    f"{state.asset}: {state.verdict()} (F {state.free_energy:+.2f}).",
                    {"asset": state.asset, "free_energy": state.free_energy},
                )

        # Level 1 - what might be true?
        ideas = self.generator.generate(features, forward_returns)
        yield self._say("IDEA", 1, f"Searched {len(features)} features, {len(ideas)} worth attacking.")
        if not ideas:
            yield self._say("VOICE", 4, "Nothing here rose above noise. No proposal.")
            return

        survivors: list[tuple[Hypothesis, RedTeamReport, float]] = []
        for h in ideas:
            yield self._say("IDEA", 1, f"{h.claim} (prior {h.prior:.2f}).", {"hypothesis": h.id})

            # Level 1 - what would kill it?
            report = self.red_team.run(h, features[h.feature], forward_returns)
            for attack in report.attacks:
                if not attack.survived:
                    yield self._say("ATTACK", 1, f"{h.id} {attack.name}: {attack.finding}",
                                    {"hypothesis": h.id, "severity": attack.severity})
            confidence = report.adjusted_confidence(h.prior)
            if not report.survived:
                yield self._say(
                    "ATTACK", 1,
                    f"{h.id} is dead - killed by {', '.join(a.name for a in report.kill_shots)}.",
                    {"hypothesis": h.id, "survived": False},
                )
                continue
            yield self._say("ATTACK", 1,
                            f"{h.id} survived all six attacks. Confidence {h.prior:.2f} -> {confidence:.2f}.",
                            {"hypothesis": h.id, "survived": True, "confidence": confidence})
            survivors.append((h, report, confidence))

        if not survivors:
            yield self._say("VOICE", 4,
                            "Every idea died under attack. That is a result, not a failure - "
                            "we just saved the money we would have lost.")
            return

        # Level 2 - what would happen if we acted?
        best, report, confidence = max(survivors, key=lambda s: s[2])
        edge_bps = best.edge_bps - self.cost_bps   # est. gross, net of assumed cost
        if graph is not None:
            mv = Multiverse(graph, outcome="pnl_bps").explore(
                {
                    "half size": {"size": requested_size / 2},
                    "as proposed": {"size": requested_size},
                    "double size": {"size": requested_size * 2},
                },
                n=2000, seed=self.seed,
            )
            for branch in mv.ranked():
                yield self._say(
                    "BRANCH", 2,
                    f"SIMULATED - {branch.label}: {branch.outcome_mean:+.1f} bps "
                    f"(regret {branch.regret_vs(mv.baseline):+.1f}).",
                    {"branch": branch.label, "simulated": True},
                )

        # Level 1 - are we allowed?
        proposal = Proposal(
            hypothesis_id=best.id,
            size_fraction=requested_size,
            confidence=confidence,
            expected_edge_bps=edge_bps,
            n_trades=best.n_obs,
            survived_red_team=report.survived,
            current_drawdown=book_drawdown,
            volatility_bps=best.vol_bps,
            book_volatility_bps=book_volatility_bps,
            book_weight=book_weight,
        )
        ruling: Ruling = self.arbiter.rule(proposal)
        for violation in ruling.violations:
            yield self._say("RULING", 1, f"{violation.rule}: {violation.message}",
                            {"hard": violation.hard})
        for note in ruling.notes:
            yield self._say("RULING", 1, note, {"note": True})
        yield self._say(
            "RULING", 1,
            f"{ruling.decision} - max size {ruling.max_size_fraction:.3f} of book"
            + (f", bound by {ruling.binding_constraint}." if not ruling.vetoed else "."),
            {"decision": ruling.decision, "size": ruling.max_size_fraction,
             "binding": ruling.binding_constraint},
        )

        # Level 2 - what did we learn, and what next?
        self.inference.believe(best.feature, mean=edge_bps, variance=max(abs(edge_bps), 1.0))
        update = self.inference.observe(Observation(best.feature, edge_bps, variance=9.0))
        yield self._say("BELIEF", 2,
                        f"Belief on '{best.feature}' now {update.belief_after.mean:+.1f} "
                        f"+/- {update.belief_after.sd:.1f} bps (surprise {update.surprise:.2f}).",
                        {"free_energy": update.free_energy})

        action, _ = self.inference.select_action([
            Action("size it", best.feature, expected_value=edge_bps, information_gain=0.1, cost=self.cost_bps),
            Action("probe small", best.feature, expected_value=edge_bps * 0.1, information_gain=0.8, cost=0.5),
            Action("stand down", best.feature, expected_value=0.0, information_gain=0.0, cost=0.0),
        ])
        yield self._say("BELIEF", 2, f"Next best experiment: {action.name}.", {"action": action.name})

        # Level 4 - say it out loud
        yield self._say("VOICE", 4, self._verdict_line(best, ruling, edge_bps, confidence, action.name))

    # ── helpers ──────────────────────────────────────────────────────────────
    def _verdict_line(self, h: Hypothesis, ruling: Ruling, edge_bps: float,
                      confidence: float, next_action: str) -> str:
        if ruling.vetoed:
            hard = "; ".join(v.message for v in ruling.violations if v.hard)
            return f"Best surviving idea was {h.id}, and risk vetoed it. {hard} No position."
        head = (
            f"{h.id} on '{h.feature}' survived the red team at {confidence:.2f} confidence, "
            f"est. {edge_bps:+.1f} bps net."
        )
        size = f" Risk permits {ruling.max_size_fraction:.1%} of the book"
        size += " (clamped)." if ruling.decision == "CLAMP" else "."
        return head + size + f" Recommended next move: {next_action}."

    def _say(self, channel: str, level: int, text: str, payload: dict | None = None) -> Utterance:
        u = Utterance(channel, level, text, payload or {})
        self.transcript.append(u)
        return u

    # ── output modes ─────────────────────────────────────────────────────────
    def run(self, *a, **kw) -> list[Utterance]:
        """Consume the whole stream and return the transcript."""
        return list(self.stream(*a, **kw))

    def speak(self, *a, **kw) -> None:
        """Print everything, telemetry included."""
        for u in self.stream(*a, **kw):
            print(u)

    def voice_only(self, *a, **kw) -> list[str]:
        """What the voice layer would actually read aloud."""
        return [u.spoken() for u in self.stream(*a, **kw) if u.channel == "VOICE"]


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5: THE OPERATIONAL INTERFACE
# ═════════════════════════════════════════════════════════════════════════════
#
# A persistent, context-aware console. Ask it a macro question in words, and it
# walks the knowledge graph for established causal routes, pulls the matching
# overnight counterfactual, streams the breakdown, and ends with exactly one
# recommendation that a human must approve before anything happens.
#
# THREE THINGS IT WILL NOT DO
#
#   It will not answer a question it has no overnight run for. "I have not
#   modelled that" is a complete answer and it gives it.
#
#   It will not walk a correlation as though it were a mechanism. If the only
#   route from the shock to the book runs through an unoriented edge, it says
#   which experiment would fix that and stops.
#
#   It will not act. Every adjustment sits behind an approval gate that only a
#   human opens. A rejection is recorded with its reason, because the reasons
#   are the most valuable data the system generates.
#
# ON "SPATIAL" AND "VOICE": the dashboard lays the book out as panels rather
# than rows, and the VOICE channel is written to be read aloud - short sentences,
# no symbols, numbers rounded to what a person can hold in their head. There is
# no 3D renderer and no speech synthesiser here. Calling a text panel "spatial"
# would be overselling it; calling the voice channel "voice" is accurate only in
# the sense that it is the text a TTS would read.


def _tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens of 3+ characters. Used on both sides of a match."""
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return {word for word in cleaned.split() if len(word) >= 3}


@dataclass
class SessionState:
    """
    What JARVIS remembers between restarts.

    Context-awareness is mostly memory plus the discipline to use it: the
    positions it is reasoning about, what it has already recommended, and - most
    valuable of all - what a human rejected and why.
    """

    operator: str = "unknown"
    positions: dict[str, float] = field(default_factory=dict)
    drawdown: float = 0.0
    queries: list[str] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    @property
    def gross(self) -> float:
        return float(sum(abs(v) for v in self.positions.values()))

    def rejections(self) -> list[dict]:
        return [d for d in self.decisions if d["decision"] == "REJECTED"]

    def previously_rejected(self, entity: str) -> list[dict]:
        """Has a human already said no to something like this? Ask before asking again."""
        return [d for d in self.rejections() if d.get("entity") == entity]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "SessionState":
        target = Path(path)
        if not target.exists():
            return cls()
        return cls(**json.loads(target.read_text()))


@dataclass
class Adjustment:
    """
    The single recommendation. One, not a menu.

    A list of five options is a way of not having an opinion. This carries one
    action, the size the arbiter permits, what the claim rests on, and - the part
    that matters - what would make it wrong.
    """

    entity: str
    action: str
    size_fraction: float
    rationale: str
    confidence: float
    evidence_basis: str                       # REAL / est. / SIMULATED
    falsifiers: tuple[str, ...] = ()
    blocked_questions: tuple[str, ...] = ()
    ruling_note: str = ""
    created: float = field(default_factory=time.time)
    decision: str = "PENDING"                 # PENDING | APPROVED | REJECTED
    decided_by: str = ""
    decided_note: str = ""

    @property
    def actionable(self) -> bool:
        return self.decision == "APPROVED" and self.size_fraction > 0

    def card(self, width: int = 76) -> str:
        """
        The approval card. Wrapped, not truncated - a falsifier cut off mid-word
        is a falsifier nobody reads, and these are the lines that matter most.
        """
        import textwrap

        inner = width - 4
        rule = "+" + "-" * (width - 2) + "+"

        def row(text: str = "", indent: int = 0) -> list[str]:
            if not text:
                return ["|" + " " * (width - 2) + "|"]
            pad = " " * indent
            return ["| " + line.ljust(inner) + " |"
                    for line in textwrap.wrap(text, inner - indent,
                                              initial_indent=pad, subsequent_indent=pad + "  ")]

        lines = [rule]
        lines += row("RECOMMENDATION")
        lines += row(self.action.upper(), indent=2)
        lines += row(f"size {self.size_fraction:.3%} of book   "
                     f"confidence {self.confidence:.2f}   basis: {self.evidence_basis}",
                     indent=2)
        lines += [rule]
        lines += row("WHY", )
        lines += row(self.rationale, indent=2)
        if self.ruling_note:
            lines += row("RISK ARBITER")
            lines += row(self.ruling_note, indent=2)
        lines += [rule]
        lines += row("WHAT WOULD MAKE THIS WRONG")
        for falsifier in self.falsifiers:
            lines += row("- " + falsifier, indent=2)
        if self.blocked_questions:
            lines += row("NOT ESTABLISHED")
            for question in self.blocked_questions:
                lines += row("? " + question, indent=2)
        lines += [rule]
        decided = (f"{self.decision}"
                   + (f" by {self.decided_by}" if self.decided_by else "")
                   + (f" - {self.decided_note}" if self.decided_note else ""))
        lines += row(decided)
        lines += [rule]
        return "\n".join(lines)


class ApprovalRequired(RuntimeError):
    """Raised when something tries to act on an unapproved adjustment."""


class JarvisInterface:
    """
    The console. Persistent, context-aware, and structurally unable to act alone.

    Query resolution is keyword matching against entity names - not language
    understanding, and it is called that in the code so nobody mistakes it for
    more. What makes the answer good is the graph and the overnight store behind
    it, not the parser in front.
    """

    def __init__(self, graph, store, session: SessionState | None = None,
                 arbiter: RiskArbiter | None = None, session_path: str | Path | None = None):
        self.graph = graph
        self.store = store
        self.session = session or SessionState()
        self.arbiter = arbiter or RiskArbiter()
        self.session_path = Path(session_path) if session_path else None
        self.transcript: list[Utterance] = []
        self.pending: Adjustment | None = None

    # ── the spatial layer ────────────────────────────────────────────────────
    def dashboard(self, width: int = 78) -> str:
        """The book laid out as panels. Ordered by what is most exposed."""
        lines = ["=" * width,
                 f" JARVIS | operator {self.session.operator} | gross {self.session.gross:.1%} "
                 f"| drawdown {self.session.drawdown:.1%}",
                 "=" * width]

        if self.session.positions:
            lines.append(" POSITIONS")
            for name, size in sorted(self.session.positions.items(),
                                     key=lambda kv: -abs(kv[1])):
                bar = ("#" if size >= 0 else "~") * max(1, int(abs(size) * 200))
                lines.append(f"   {name:<18} {size:+7.3f}  {bar}")

        exposed = self._most_exposed()
        if exposed:
            lines += ["", " EXPOSURE BY CAUSAL DISTANCE (established routes only)"]
            for entity_id, paths in exposed[:5]:
                best = paths[0]
                lines.append(f"   {self.graph.name_of(entity_id):<28} "
                             f"{len(paths)} route(s), strongest effect {best.effect:+.2f} "
                             f"at confidence {best.confidence:.2f}")

        coverage = self.store.coverage()
        lines += ["", " OVERNIGHT COVERAGE"]
        if coverage:
            for entity_id, count in sorted(coverage.items()):
                lines.append(f"   {self.graph.name_of(entity_id):<28} {count} scenario(s) ready")
        else:
            lines.append("   nothing precomputed - every question will be answered 'not modelled'")

        blocked = self.graph.ambiguous_edges()
        if blocked:
            lines += ["", f" NOT ESTABLISHED ({len(blocked)} edge(s) of unknown direction)"]
            for edge in blocked[:3]:
                lines.append(f"   {self.graph.name_of(edge.source)} -- "
                             f"{self.graph.name_of(edge.target)}  (needs an intervention)")

        rejections = self.session.rejections()
        if rejections:
            lines += ["", f" STANDING HUMAN DECISIONS ({len(rejections)} rejection(s) on file)"]
            for decision in rejections[-3:]:
                lines.append(f"   {decision.get('entity', '?'):<20} rejected by "
                             f"{decision.get('by', '?')}: {decision.get('note', '')[:34]}")
        lines.append("=" * width)
        return "\n".join(lines)

    def _most_exposed(self):
        book = next((e.entity_id for e in self.graph.of_kind("book")), None)
        if book is None:
            return []
        out = []
        for entity in self.graph.entities.values():
            if entity.entity_id == book:
                continue
            paths = self.graph.causal_paths(entity.entity_id, target=book)
            if paths:
                out.append((entity.entity_id, paths))
        return sorted(out, key=lambda pair: abs(pair[1][0].effect) * pair[1][0].confidence,
                      reverse=True)

    # ── the query ────────────────────────────────────────────────────────────
    def resolve(self, query: str) -> str | None:
        """
        Keyword match against entity names. Not language understanding.

        Deliberately dumb and deliberately transparent: it returns None rather
        than its best guess, because a confidently misresolved entity produces a
        confident answer to a question nobody asked.
        """
        words = _tokens(query)
        best, best_score = None, 0
        for entity in self.graph.entities.values():
            # Tokenise both sides IDENTICALLY. They did not match before, so a
            # query naming "USD/JPY" resolved to nothing and the interesting
            # refusal - a real relation whose direction was never established -
            # never got a chance to fire.
            score = len(words & (_tokens(entity.name) | _tokens(entity.entity_id)))
            if score > best_score:
                best, best_score = entity.entity_id, score
        return best

    def ask(self, query: str, max_paths: int = 4) -> Iterator[Utterance]:
        """
        Answer a macro question. Streams the reasoning, ends with one adjustment.

        The order is the argument: what was asked, how it reaches us, what we
        precomputed, what risk permits, and only then what to do.
        """
        self.session.queries.append(query)
        yield self._say("SCENARIO", 4, f'Query: "{query}"')

        entity_id = self.resolve(query)
        if entity_id is None:
            yield self._say("VOICE", 4,
                            "I do not have an entity matching that query. Nothing in my "
                            "world model corresponds to it, so I have no answer to give.")
            return

        entity = self.graph.entities[entity_id]
        yield self._say("SCENARIO", 4, f"Resolved to: {entity} "
                                       f"(keyword match, not language understanding)")

        prior = self.session.previously_rejected(entity_id)
        if prior:
            yield self._say("SCENARIO", 4,
                            f"Context: you rejected a recommendation on this "
                            f"{len(prior)} time(s) before. Most recent reason: "
                            f"{prior[-1].get('note', 'none given')}")

        # -- how does it reach us? --
        book = next((e.entity_id for e in self.graph.of_kind("book")), None)
        paths = self.graph.causal_paths(entity_id, target=book) if book else []
        if not paths:
            yield self._say("PATH", 4,
                            "No established causal route from there to the book.")
            for question in self.graph.blocked_questions(entity_id):
                yield self._say("PATH", 4, f"Not established: {question}")
            yield self._say("VOICE", 4,
                            "There is no established causal route from that to our book. "
                            "There may be a real one, but nothing has oriented it, so I "
                            "will not guess at a direction. No recommendation.")
            return

        for path in paths[:max_paths]:
            yield self._say("PATH", 4, path.describe(self.graph),
                            {"effect": path.effect, "confidence": path.confidence})

        # -- what did we precompute? --
        results = self.store.for_entity(entity_id)
        if not results:
            yield self._say("COUNTERFACTUAL", 4,
                            f"No overnight run covers {entity.name}.")
            yield self._say("VOICE", 4,
                            f"I can see how {entity.name} reaches the book, but I have not "
                            f"run the counterfactual. Queue it tonight and ask me tomorrow. "
                            f"I am not going to improvise a number.")
            return

        book_hash = book_state_hash(self.session.positions)
        usable = []
        for result in results:
            stale = result.stale_against(book_hash)
            if stale:
                yield self._say("COUNTERFACTUAL", 4,
                                f"SKIPPED {result.name}: {stale}", {"stale": True})
                continue
            usable.append(result)
            yield self._say("COUNTERFACTUAL", 4,
                            f"{SIMULATED} - {result.name}: baseline "
                            f"{result.baseline_mean:+.1f} bps over {result.samples:,} paths",
                            {"simulated": True})
            for branch in sorted(result.branches, key=lambda b: b.outcome_mean, reverse=True):
                yield self._say("COUNTERFACTUAL", 4,
                                f"{SIMULATED} -   {branch.label:<18} "
                                f"{branch.outcome_mean:+7.1f} bps "
                                f"(p05 {branch.outcome_p05:+.0f}, p95 {branch.outcome_p95:+.0f})",
                                {"simulated": True})

        if not usable:
            yield self._say("VOICE", 4,
                            "Every counterfactual I hold for that is stale against the "
                            "current book. I would rather give you nothing than a number "
                            "computed for a portfolio we no longer have.")
            return

        # -- what does risk permit? --
        adjustment = self._build_adjustment(entity, paths, usable)
        for line in adjustment.ruling_note.split(" | "):
            yield self._say("RULING", 1, line)

        if adjustment.size_fraction <= 0:
            yield self._say("VOICE", 4,
                            f"The model likes it, and risk does not. {adjustment.ruling_note} "
                            f"No position. I am not going to argue with the arbiter.")
            return

        self.pending = adjustment
        yield self._say("ADJUSTMENT", 4, adjustment.rationale,
                        {"size": adjustment.size_fraction, "pending": True})
        yield self._say("APPROVAL", 4,
                        "Awaiting your approval. Nothing happens until you give it.")
        yield self._say("VOICE", 4, self._spoken(adjustment))

    def _build_adjustment(self, entity, paths, results) -> Adjustment:
        best_path = paths[0]
        result = max(results, key=lambda r: abs(r.best.outcome_mean - r.baseline_mean))
        best_branch = result.best

        # Simulated evidence sizes nothing, so the size claim rests on the
        # established causal path's confidence, and the arbiter rules on that.
        edge_bps = abs(best_branch.outcome_mean - result.baseline_mean)
        requested = min(0.05, self.session.gross * 0.5) or 0.02
        ruling = self.arbiter.rule(Proposal(
            hypothesis_id=f"jarvis:{entity.entity_id}",
            size_fraction=requested,
            confidence=best_path.confidence,
            expected_edge_bps=edge_bps,
            n_trades=250,
            survived_red_team=True,
            current_drawdown=self.session.drawdown,
            volatility_bps=max(best_branch.spread / 3.29, 1.0),   # p05-p95 is ~3.29 sigma
        ))
        note = (f"{ruling.decision} at {ruling.max_size_fraction:.4f}"
                + (f" (bound by {ruling.binding_constraint})" if not ruling.vetoed else "")
                + (" | " + "; ".join(v.message for v in ruling.violations if v.hard)
                   if ruling.vetoed else ""))

        action = f"{best_branch.label.replace('shock, ', '')} the {entity.name} exposure"
        return Adjustment(
            entity=entity.entity_id,
            action=action,
            size_fraction=0.0 if ruling.vetoed else ruling.max_size_fraction,
            rationale=(f"{entity.name} reaches the book via "
                       f"{' -> '.join(self.graph.name_of(n) for n in best_path.nodes)}; "
                       f"under that shock the best response was '{best_branch.label}' at "
                       f"{best_branch.outcome_mean:+.0f} bps {SIMULATED}"),
            confidence=best_path.confidence,
            evidence_basis=f"causal path {best_path.weakest_evidence}, outcome {SIMULATED}",
            falsifiers=(
                f"the {self.graph.name_of(best_path.nodes[1]) if len(best_path.nodes) > 1 else '?'}"
                f" link is {best_path.weakest_evidence.lower()} - if that arrow is wrong, "
                f"so is this",
                f"the counterfactual assumed the book as of {result.book_state_hash}; "
                f"trade anything material and it is void",
                f"outcome spread is {best_branch.spread:.0f} bps wide - the direction is "
                f"a better claim than the magnitude",
            ),
            blocked_questions=tuple(self.graph.blocked_questions(entity.entity_id)),
            ruling_note=note,
        )

    def _spoken(self, adjustment: Adjustment) -> str:
        """Written to be read aloud: short sentences, no symbols, round numbers."""
        return (f"{self.graph.name_of(adjustment.entity)}. "
                f"One recommendation: {adjustment.action}, "
                f"about {adjustment.size_fraction * 100:.1f} percent of the book. "
                f"Confidence {adjustment.confidence:.0%}, and the outcome number is simulated. "
                f"The weakest part is the {adjustment.evidence_basis.split(',')[0]}. "
                f"Say approve or reject.")

    # ── the gate ─────────────────────────────────────────────────────────────
    def approve(self, operator: str, note: str = "") -> Adjustment:
        """A human, named, on the record. Nothing else opens this gate."""
        if self.pending is None:
            raise ApprovalRequired("there is no pending adjustment to approve")
        self.pending.decision = "APPROVED"
        self.pending.decided_by = operator
        self.pending.decided_note = note
        self._record(self.pending)
        approved, self.pending = self.pending, None
        return approved

    def reject(self, operator: str, note: str = "") -> Adjustment:
        """
        Rejection is data, not friction.

        The reasons a human says no are the most valuable output this system
        produces, and they are kept and surfaced the next time the same entity
        comes up.
        """
        if self.pending is None:
            raise ApprovalRequired("there is no pending adjustment to reject")
        self.pending.decision = "REJECTED"
        self.pending.decided_by = operator
        self.pending.decided_note = note
        self._record(self.pending)
        rejected, self.pending = self.pending, None
        return rejected

    def execute(self, adjustment: Adjustment) -> Intent:
        """
        Turn an APPROVED adjustment into an intent. Still not an order.

        Two gates, not one: a human approves, and even then what comes out is an
        intent that some other system, outside this tree, would have to act on.
        """
        if not adjustment.actionable:
            raise ApprovalRequired(
                f"adjustment is {adjustment.decision}; only an APPROVED one becomes an intent")
        return Intent("bid", adjustment.size_fraction,
                      f"approved by {adjustment.decided_by}", adjustment.entity)

    def _record(self, adjustment: Adjustment) -> None:
        self.session.decisions.append({
            "entity": adjustment.entity, "action": adjustment.action,
            "size": adjustment.size_fraction, "decision": adjustment.decision,
            "by": adjustment.decided_by, "note": adjustment.decided_note,
            "at": time.time(),
        })
        if self.session_path:
            self.session.save(self.session_path)

    def _say(self, channel: str, level: int, text: str, payload: dict | None = None) -> Utterance:
        utterance = Utterance(channel, level, text, payload or {})
        self.transcript.append(utterance)
        return utterance

    def speak(self, query: str) -> None:
        for utterance in self.ask(query):
            print(utterance)

    def voice_only(self, query: str) -> list[str]:
        return [u.text for u in self.ask(query) if u.channel == "VOICE"]


def _demo_graph() -> CausalGraph:
    return (
        CausalGraph()
        .add("liquidity", (), lambda v, e: 1.0 + 0.3 * e)
        .add("size", (), lambda v, e: np.full_like(e, 0.05))
        .add("impact_bps", ("size", "liquidity"),
             lambda v, e: 40.0 * v["size"] / np.maximum(v["liquidity"], 0.2) + 0.5 * np.abs(e))
        .add("alpha_bps", (), lambda v, e: 9.0 + 3.0 * e)
        .add("pnl_bps", ("alpha_bps", "impact_bps"),
             lambda v, e: v["alpha_bps"] - v["impact_bps"] + 0.2 * e)
    )


if __name__ == "__main__":
    from tensors.market_tensors import synth_panel

    rng = np.random.default_rng(7)
    n = 800
    real = rng.normal(0, 1, n)
    features = {
        "order_flow_imbalance": real,
        "rsi_14": rng.normal(0, 1, n),
        "twitter_sentiment": rng.normal(0, 1, n),
        "moon_phase": rng.normal(0, 1, n),
    }
    fwd = (0.14 * real + rng.normal(0, 1, n)) * 0.01

    tensor = MarketTensor.from_panels({
        "ES": synth_panel(vol=0.006, shock=0.010, seed=3),
        "NQ": synth_panel(vol=0.020, seed=4),
    })

    print("=" * 78)
    print(" STARK MK18 - jarvis node")
    print("=" * 78 + "\n")

    jarvis = JarvisNode(seed=1, cost_bps=5.0)
    jarvis.speak(features, fwd, tensor=tensor, graph=_demo_graph(), book_drawdown=0.03)

    print("\n" + "-" * 78)
    print(" voice channel only:")
    for line in JarvisNode(seed=1).voice_only(features, fwd, tensor=tensor):
        print(" ", line)

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 5: the operational interface
    # ═════════════════════════════════════════════════════════════════════════
    import tempfile

    print("\n\n" + "=" * 78)
    print(" PHASE 5 - JARVIS OPERATIONAL INTERFACE")
    print("=" * 78)

    knowledge = demo_graph()
    positions = {"crude": 0.06, "diesel_crack": 0.03}
    session = SessionState(operator="josh", positions=positions, drawdown=0.04)
    store = OvernightRunner(demo_model(), samples=4_000, seed=11).run_all(
        DEFAULT_SCENARIOS, positions)

    with tempfile.TemporaryDirectory() as tmp:
        session_file = Path(tmp) / "session.json"
        jarvis = JarvisInterface(knowledge, store, session, session_path=session_file)

        print("\n" + jarvis.dashboard())

        print("\n\n--- the question ---\n")
        query = ("What is our exposure to a sudden Strait of Malacca supply disruption "
                 "and how do we position for it?")
        for utterance in jarvis.ask(query):
            print(utterance)

        if jarvis.pending:
            print("\n" + jarvis.pending.card())

            print("\n--- sovereign approval: a human, named, on the record ---")
            rejected = jarvis.reject("josh", "freight leg is one analogue, not a model")
            print(f"  {rejected.decision} by {rejected.decided_by}: {rejected.decided_note}")
            try:
                jarvis.execute(rejected)
            except ApprovalRequired as e:
                print(f"  execute() refused -> {e}")

            print("\n--- ask the same thing again; it remembers ---")
            for utterance in jarvis.ask(query):
                if utterance.channel == "SCENARIO" and "rejected" in utterance.text:
                    print(f"  {utterance}")
            approved = jarvis.approve("josh", "second look, freight leg re-checked")
            print(f"\n  {approved.decision} by {approved.decided_by}")
            print(f"  becomes: {jarvis.execute(approved)}")
            print("  ...an INTENT. Still not an order. Execution lives outside this tree.")

        print("\n\n--- three questions it refuses, and why ---\n")
        print("1. something not in the world model:")
        for line in jarvis.voice_only("What about a solar flare hitting the exchange?"):
            print(f"   {line}")

        print("\n2. related, but the direction was never established:")
        for utterance in jarvis.ask("What does USD/JPY do to us?"):
            if utterance.channel in ("PATH", "VOICE"):
                print(f"   {utterance.text}")

        print("\n3. reachable, but never modelled overnight:")
        empty = JarvisInterface(knowledge, OvernightStore(), session)
        for line in empty.voice_only(query):
            print(f"   {line}")

        print("\n" + "=" * 78)
        print(" Every number above that came from a model said SIMULATED. Every causal")
        print(" step named the evidence that oriented it. The one recommendation carried")
        print(" what would make it wrong, and a human had to say yes before it became")
        print(" even an intent - let alone an order.")
        print("=" * 78)
