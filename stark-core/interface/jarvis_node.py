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

  channel   what it is                     for
  ---------------------------------------------------------------
  FIELD     tensor readings                the spatial layer (where to look)
  IDEA      hypotheses                     the neural layer
  ATTACK    red team findings              the adversarial layer
  RULING    risk arbiter decisions         the hard layer - never editorialised
  BRANCH    causal simulations             the counterfactual layer
  BELIEF    active inference updates       the learning layer
  VOICE     plain-English summary          what actually gets spoken

Rule: JARVIS may reformat anything and contradict nothing. If it ever finds
itself softening a VETO, that is a bug in this file, not a feature.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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

CHANNELS = ("FIELD", "IDEA", "ATTACK", "RULING", "BRANCH", "BELIEF", "VOICE")


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
        edge_bps = self._edge_bps(features[best.feature], forward_returns) - self.cost_bps
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
        )
        ruling: Ruling = self.arbiter.rule(proposal)
        for violation in ruling.violations:
            yield self._say("RULING", 1, f"{violation.rule}: {violation.message}",
                            {"hard": violation.hard})
        for note in ruling.notes:
            yield self._say("RULING", 1, note, {"note": True})
        yield self._say("RULING", 1,
                        f"{ruling.decision} - max size {ruling.max_size_fraction:.3f} of book.",
                        {"decision": ruling.decision, "size": ruling.max_size_fraction})

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

    def _edge_bps(self, values: Sequence[float], fwd: Sequence[float]) -> float:
        """Same crude estimator the red team's cost attack uses. Est., and labelled so."""
        from core.generator import rank_ic
        return float(abs(rank_ic(np.asarray(values, float), np.asarray(fwd, float)))
                     * float(np.std(fwd)) * 10_000.0)

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
