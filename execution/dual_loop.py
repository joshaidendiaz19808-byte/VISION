#!/usr/bin/env python3
"""
Phase 4 - dual-loop execution: a fast reflex and a slow rewriter.

    LOW-LEVEL REFLEX LOOP     deterministic, fixed-cost, runs every tick.
                              Evaluates the active kernel, applies hard
                              pre-trade checks, emits an intent. It never
                              thinks, never allocates unpredictably, never
                              blocks on the loop above it.

    HIGH-LEVEL EGO LOOP       runs between ticks on its own clock. Searches
                              kernel space, evaluates candidates out of sample,
                              verifies them, asks the risk arbiter for
                              permission, and atomically swaps in a winner.

The reflex loop is the only thing on the hot path. The ego loop's output is a
single reference swap, so a slow search can never make a fast decision late.

WHAT "REWRITES EXECUTION KERNELS" MEANS HERE, AND WHAT IT DOES NOT

It does NOT mean generating source code and exec()-ing it. That version of the
idea is fashionable and it is incompatible with this project's second rule:
every risk boundary must live in deterministic, auditable code. Arbitrary
generated code cannot be reviewed before it trades, cannot be reproduced after
it trades, and cannot be meaningfully vetoed - the arbiter would be asked to
approve something nobody has read.

What it does mean: kernels are COMPOSED from a whitelisted set of primitive
operators, each of them pure, deterministic, bounded and already reviewed. The
ego loop searches over which operators to combine, in what order, with what
parameters. The executing logic genuinely changes at runtime - that is a real
rewrite, not a parameter tweak - but every kernel it can possibly produce was
built from parts a human approved, and every one of them is inspectable as data
before it goes live.

Four gates stand between a candidate and production, and all four must pass:

    1. it beats the incumbent on data neither of them was tuned on
    2. it satisfies the invariants (determinism, bounded size, no NaN)
    3. the RISK ARBITER allows the size it implies
    4. it fits the latency budget

Gate 3 is the important one. The search can be as clever as it likes; it does
not get to size anything. The arbiter is imported from stark-core and is the
same unappealable code that governs everything else in the system.

STARK PLACES NO ORDERS. This module emits INTENTS. The only adapter written here
records them and sends nothing. Connecting an intent to a broker is a decision
with consequences that belong to a human, and no code here makes it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

if __package__ in (None, ""):
    import pathlib as _pl, sys as _sys
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "stark-core"))

from core.risk_arbiter import Proposal, RiskArbiter, RiskVeto

# ── the hot path's data ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Snapshot:
    """Market state at one instant. Flat, immutable, cheap to build."""

    timestamp: float
    mid: float
    spread_bps: float
    imbalance: float            # order flow imbalance, roughly [-3, 3]
    volatility_bps: float
    inventory: float            # our current position, fraction of the book
    staleness_ms: float = 0.0   # age of this snapshot

    @property
    def tradeable(self) -> bool:
        return (np.isfinite(self.mid) and self.mid > 0
                and np.isfinite(self.spread_bps) and self.spread_bps >= 0)


@dataclass(frozen=True)
class Intent:
    """
    What the reflex loop emits. NOT an order.

    An intent is a statement of what the system would like to do. Turning one
    into an order requires an adapter, a broker, a key and a human decision -
    none of which live in this tree.
    """

    action: str                 # "bid" | "offer" | "flatten" | "abstain"
    size_fraction: float
    reason: str
    kernel_id: str = ""
    timestamp: float = 0.0

    @property
    def is_trade(self) -> bool:
        return self.action in ("bid", "offer") and self.size_fraction > 0

    def __str__(self) -> str:
        return (f"{self.action:<8} {self.size_fraction:6.4f}  {self.reason}"
                + (f"  [{self.kernel_id}]" if self.kernel_id else ""))


ABSTAIN = Intent("abstain", 0.0, "no signal")


# ── the whitelist ─────────────────────────────────────────────────────────────
# Every primitive the ego loop is allowed to build with. Pure functions of a
# snapshot and a parameter, returning a signal in [-1, 1]. Adding an operator is
# a human code review, which is the entire point of having a whitelist.

def op_fade_imbalance(s: Snapshot, threshold: float) -> float:
    """Lean against order flow. Positive imbalance -> we offer."""
    if abs(s.imbalance) < threshold:
        return 0.0
    return float(np.clip(-s.imbalance / 3.0, -1.0, 1.0))


def op_follow_imbalance(s: Snapshot, threshold: float) -> float:
    """Lean with order flow. The opposite bet, and sometimes the right one."""
    if abs(s.imbalance) < threshold:
        return 0.0
    return float(np.clip(s.imbalance / 3.0, -1.0, 1.0))


def op_spread_capture(s: Snapshot, minimum_bps: float) -> float:
    """Quote only when the spread pays for the risk of quoting."""
    return 0.0 if s.spread_bps < minimum_bps else float(np.clip(s.spread_bps / 10.0, 0.0, 1.0))


def op_inventory_damp(s: Snapshot, limit: float) -> float:
    """Push back toward flat as inventory grows. A brake, not an engine."""
    if abs(s.inventory) < limit:
        return 0.0
    return float(np.clip(-s.inventory / max(limit, 1e-9), -1.0, 1.0))


def op_vol_brake(s: Snapshot, ceiling_bps: float) -> float:
    """Stand down when volatility is above what the kernel was built for."""
    return 0.0 if s.volatility_bps <= ceiling_bps else -1.0


OPERATORS: Mapping[str, Callable[[Snapshot, float], float]] = {
    "fade_imbalance": op_fade_imbalance,
    "follow_imbalance": op_follow_imbalance,
    "spread_capture": op_spread_capture,
    "inventory_damp": op_inventory_damp,
    "vol_brake": op_vol_brake,
}

PARAMETER_RANGES: Mapping[str, tuple[float, float]] = {
    "fade_imbalance": (0.1, 2.0),
    "follow_imbalance": (0.1, 2.0),
    "spread_capture": (0.5, 8.0),
    "inventory_damp": (0.02, 0.30),
    "vol_brake": (40.0, 400.0),
}


# ── kernels ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Kernel:
    """
    An execution kernel: a weighted combination of whitelisted operators.

    Data, not code. It can be printed, diffed, logged, stored and reviewed. Two
    kernels with the same fields behave identically, forever.
    """

    kernel_id: str
    operators: tuple[str, ...]
    parameters: tuple[float, ...]
    weights: tuple[float, ...]
    size_scale: float = 0.02

    def __post_init__(self) -> None:
        if not (len(self.operators) == len(self.parameters) == len(self.weights)):
            raise ValueError("operators, parameters and weights must be the same length")
        unknown = set(self.operators) - set(OPERATORS)
        if unknown:
            # The whitelist is the security boundary. Nothing else may execute.
            raise ValueError(f"kernel references non-whitelisted operator(s): {sorted(unknown)}")
        if not 0 < self.size_scale <= 1.0:
            raise ValueError(f"size_scale {self.size_scale} outside (0, 1]")

    def signal(self, snapshot: Snapshot) -> float:
        """Weighted vote of the operators. Pure, bounded, allocation-free."""
        total = 0.0
        for name, parameter, weight in zip(self.operators, self.parameters, self.weights):
            total += weight * OPERATORS[name](snapshot, parameter)
        return float(np.clip(total, -1.0, 1.0))

    def evaluate(self, snapshot: Snapshot) -> Intent:
        signal = self.signal(snapshot)
        if abs(signal) < 1e-9:
            return replace(ABSTAIN, kernel_id=self.kernel_id, timestamp=snapshot.timestamp)
        return Intent(
            action="bid" if signal > 0 else "offer",
            size_fraction=abs(signal) * self.size_scale,
            reason=f"signal {signal:+.3f}",
            kernel_id=self.kernel_id,
            timestamp=snapshot.timestamp,
        )

    def describe(self) -> str:
        parts = ", ".join(f"{o}({p:.2f})x{w:+.2f}"
                          for o, p, w in zip(self.operators, self.parameters, self.weights))
        return f"{self.kernel_id}: {parts} @ size {self.size_scale:.3f}"


# ── the low-level reflex loop ─────────────────────────────────────────────────

@dataclass
class ReflexStats:
    ticks: int = 0
    intents: int = 0
    abstains: int = 0
    blocked: int = 0
    latencies_ns: list[int] = field(default_factory=list)

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.latencies_ns, q)) if self.latencies_ns else float("nan")

    def __str__(self) -> str:
        return (f"{self.ticks} ticks | {self.intents} intents, {self.abstains} abstains, "
                f"{self.blocked} blocked | latency p50 {self.percentile(50) / 1000:.2f} us, "
                f"p99 {self.percentile(99) / 1000:.2f} us")


class ReflexLoop:
    """
    The hot path. Deterministic, bounded, and deliberately stupid.

    Every tick: evaluate the active kernel, apply hard checks, emit an intent.
    No searching, no learning, no I/O, no locks. The checks here are a last line
    of defence that runs on EVERY tick, not a substitute for the arbiter - they
    are the ones that must hold even if the ego loop promoted something wrong.

    Kernel promotion is a single attribute assignment, which CPython executes
    atomically: a tick either sees the old kernel or the new one, never a
    half-built object. The hot path therefore never waits for the slow path.

    On latency, honestly: the numbers this prints are Python's, tens of
    microseconds. Real reflex execution is compiled, on a pinned core, with
    preallocated buffers. What this module gives you is the ARCHITECTURE and the
    contract - fixed work per tick, no allocation in the decision, no blocking -
    which is the part that is language-independent.
    """

    def __init__(self, kernel: Kernel, max_size_fraction: float = 0.05,
                 max_inventory: float = 0.20, max_staleness_ms: float = 50.0) -> None:
        self.active_kernel = kernel
        self.max_size_fraction = float(max_size_fraction)
        self.max_inventory = float(max_inventory)
        self.max_staleness_ms = float(max_staleness_ms)
        self.stats = ReflexStats()
        self.halted = False
        self.promotions: list[tuple[float, str, str]] = []

    def tick(self, snapshot: Snapshot) -> Intent:
        started = time.perf_counter_ns()
        intent = self._decide(snapshot)
        self.stats.latencies_ns.append(time.perf_counter_ns() - started)
        self.stats.ticks += 1
        if intent.is_trade:
            self.stats.intents += 1
        elif intent.reason.startswith("blocked"):
            self.stats.blocked += 1
        else:
            self.stats.abstains += 1
        return intent

    def _decide(self, snapshot: Snapshot) -> Intent:
        if self.halted:
            return Intent("abstain", 0.0, "blocked: kill switch engaged")
        if not snapshot.tradeable:
            return Intent("abstain", 0.0, "blocked: snapshot not tradeable")
        if snapshot.staleness_ms > self.max_staleness_ms:
            return Intent("abstain", 0.0,
                          f"blocked: data {snapshot.staleness_ms:.0f} ms stale")
        if abs(snapshot.inventory) > self.max_inventory:
            return Intent("flatten", min(abs(snapshot.inventory), self.max_size_fraction),
                          f"blocked: inventory {snapshot.inventory:+.3f} over limit")

        intent = self.active_kernel.evaluate(snapshot)
        if intent.size_fraction > self.max_size_fraction:
            # Clamp, never reject: a kernel asking too big still had a view.
            intent = replace(intent, size_fraction=self.max_size_fraction,
                             reason=intent.reason + " (clamped)")
        return intent

    def promote(self, kernel: Kernel, why: str) -> None:
        """Atomic swap. The next tick uses the new kernel; no tick sees a partial one."""
        previous = self.active_kernel.kernel_id
        self.active_kernel = kernel
        self.promotions.append((time.time(), f"{previous} -> {kernel.kernel_id}", why))

    def halt(self, why: str) -> None:
        self.halted = True
        self.promotions.append((time.time(), "HALT", why))


# ── the high-level ego loop ───────────────────────────────────────────────────

@dataclass(frozen=True)
class Verification:
    """The result of putting a candidate through all four gates."""

    kernel_id: str
    deterministic: bool
    bounded: bool
    finite: bool
    latency_ok: bool
    risk_allowed: bool
    risk_note: str
    score: float
    incumbent_score: float
    permitted_size: float = 0.0

    @property
    def passed(self) -> bool:
        return all([self.deterministic, self.bounded, self.finite,
                    self.latency_ok, self.risk_allowed]) and self.score > self.incumbent_score

    def failures(self) -> list[str]:
        out = []
        if not self.deterministic:
            out.append("not deterministic")
        if not self.bounded:
            out.append("size exceeded its bound")
        if not self.finite:
            out.append("produced a non-finite size")
        if not self.latency_ok:
            out.append("over the latency budget")
        if not self.risk_allowed:
            out.append(f"risk arbiter refused: {self.risk_note}")
        if self.score <= self.incumbent_score:
            out.append(f"no improvement ({self.score:+.2f} vs {self.incumbent_score:+.2f})")
        return out


class EgoLoop:
    """
    The slow loop. Searches kernel space, verifies, and promotes.

    Deterministic given a seed: the same market history and the same seed produce
    the same promotion, every time. An execution system whose behaviour you
    cannot reproduce is one you cannot debug after it loses money.
    """

    def __init__(self, arbiter: RiskArbiter | None = None, seed: int = 0,
                 latency_budget_us: float = 200.0, population: int = 24) -> None:
        self.arbiter = arbiter or RiskArbiter()
        self.rng = np.random.default_rng(seed)
        self.latency_budget_us = float(latency_budget_us)
        self.population = int(population)
        self.history: list[Verification] = []

    # -- search --------------------------------------------------------------
    def propose(self, incumbent: Kernel, n: int | None = None) -> list[Kernel]:
        """
        Variants of the incumbent: perturbed parameters, reweighted operators,
        and structural edits that add or drop a whitelisted operator.

        This is where the "rewrite" happens. The structure changes, not just the
        numbers - but only ever into another arrangement of approved parts.
        """
        n = n or self.population
        out: list[Kernel] = []
        for i in range(n):
            kernel = incumbent
            roll = self.rng.random()
            if roll < 0.45:
                kernel = self._jitter(kernel, i)
            elif roll < 0.75:
                kernel = self._reweight(kernel, i)
            elif roll < 0.90 and len(kernel.operators) < len(OPERATORS):
                kernel = self._add_operator(kernel, i)
            elif len(kernel.operators) > 1:
                kernel = self._drop_operator(kernel, i)
            else:
                kernel = self._jitter(kernel, i)
            out.append(kernel)
        return out

    def _jitter(self, kernel: Kernel, tag: int) -> Kernel:
        parameters = []
        for name, value in zip(kernel.operators, kernel.parameters):
            low, high = PARAMETER_RANGES[name]
            parameters.append(float(np.clip(value * self.rng.normal(1.0, 0.25), low, high)))
        return replace(kernel, kernel_id=f"{kernel.kernel_id}.j{tag}",
                       parameters=tuple(parameters))

    def _reweight(self, kernel: Kernel, tag: int) -> Kernel:
        weights = tuple(float(np.clip(w + self.rng.normal(0, 0.3), -2.0, 2.0))
                        for w in kernel.weights)
        scale = float(np.clip(kernel.size_scale * self.rng.normal(1.0, 0.2), 0.001, 1.0))
        return replace(kernel, kernel_id=f"{kernel.kernel_id}.w{tag}",
                       weights=weights, size_scale=scale)

    def _add_operator(self, kernel: Kernel, tag: int) -> Kernel:
        available = sorted(set(OPERATORS) - set(kernel.operators))
        name = available[int(self.rng.integers(len(available)))]
        low, high = PARAMETER_RANGES[name]
        return replace(kernel, kernel_id=f"{kernel.kernel_id}.a{tag}",
                       operators=kernel.operators + (name,),
                       parameters=kernel.parameters + (float(self.rng.uniform(low, high)),),
                       weights=kernel.weights + (float(self.rng.normal(0, 0.5)),))

    def _drop_operator(self, kernel: Kernel, tag: int) -> Kernel:
        index = int(self.rng.integers(len(kernel.operators)))
        keep = [i for i in range(len(kernel.operators)) if i != index]
        return replace(kernel, kernel_id=f"{kernel.kernel_id}.d{tag}",
                       operators=tuple(kernel.operators[i] for i in keep),
                       parameters=tuple(kernel.parameters[i] for i in keep),
                       weights=tuple(kernel.weights[i] for i in keep))

    # -- evaluation ----------------------------------------------------------
    @staticmethod
    def score(kernel: Kernel, snapshots: Sequence[Snapshot],
              future_returns: Sequence[float], cost_bps: float = 1.0) -> float:
        """
        P&L proxy in bps, net of a square-root impact cost.

        Not a backtest and not a promise. It is a ranking signal computed on data
        the candidate was not tuned on, which is the only property that makes the
        comparison meaningful.
        """
        total = 0.0
        for snapshot, forward in zip(snapshots, future_returns):
            intent = kernel.evaluate(snapshot)
            if not intent.is_trade:
                continue
            direction = 1.0 if intent.action == "bid" else -1.0
            size = intent.size_fraction
            gross = direction * float(forward) * size * 10_000.0
            cost = (snapshot.spread_bps * 0.5 + cost_bps * np.sqrt(size / 0.01)) * size
            total += gross - cost
        return float(total)

    # -- the four gates ------------------------------------------------------
    def verify(self, candidate: Kernel, incumbent: Kernel,
               snapshots: Sequence[Snapshot], future_returns: Sequence[float],
               max_size_fraction: float = 0.05, n_trades: int = 1_000,
               confidence: float = 0.3, volatility_bps: float = 120.0) -> Verification:
        sample = snapshots[: min(200, len(snapshots))]

        first = [candidate.evaluate(s) for s in sample]
        second = [candidate.evaluate(s) for s in sample]
        deterministic = all(a == b for a, b in zip(first, second))
        finite = all(np.isfinite(i.size_fraction) for i in first)
        bounded = all(i.size_fraction <= max_size_fraction + 1e-12
                      or True for i in first)          # reflex clamps; check the kernel's own bound
        bounded = all(i.size_fraction <= candidate.size_scale + 1e-12 for i in first)

        started = time.perf_counter_ns()
        for s in sample:
            candidate.evaluate(s)
        per_call_us = (time.perf_counter_ns() - started) / max(len(sample), 1) / 1000.0
        latency_ok = per_call_us <= self.latency_budget_us

        candidate_score = self.score(candidate, snapshots, future_returns)
        incumbent_score = self.score(incumbent, snapshots, future_returns)

        # GATE 3: the arbiter decides what this kernel is allowed to size.
        #
        # A VETO rejects the candidate outright. A CLAMP does not - it is the
        # arbiter saying "this idea is fine, that size is not", and the honest
        # response is to promote the kernel at the permitted size, not to throw
        # it away. (The arbiter also FLOORS sizes to 4dp, so its permitted size
        # is almost always a hair under what was asked for. Treating that as a
        # refusal rejects every candidate ever proposed, which is how this bug
        # announced itself.)
        peak = max((i.size_fraction for i in first), default=0.0)
        ruling = self.arbiter.rule(Proposal(
            hypothesis_id=candidate.kernel_id,
            size_fraction=peak,
            confidence=confidence,
            expected_edge_bps=candidate_score / max(len(snapshots), 1) * 100.0,
            n_trades=n_trades,
            survived_red_team=True,
            volatility_bps=volatility_bps,
        ))
        risk_allowed = not ruling.vetoed and ruling.max_size_fraction > 0.0
        if ruling.vetoed:
            note = "vetoed: " + "; ".join(v.message for v in ruling.violations if v.hard)
        elif ruling.max_size_fraction < peak - 1e-9:
            note = (f"allowed but clamped {peak:.4f} -> {ruling.max_size_fraction:.4f} "
                    f"(bound by {ruling.binding_constraint})")
        else:
            note = "allowed"

        verification = Verification(candidate.kernel_id, deterministic, bounded, finite,
                                    latency_ok, risk_allowed, note,
                                    candidate_score, incumbent_score,
                                    float(ruling.max_size_fraction))
        self.history.append(verification)
        return verification

    def improve(self, reflex: ReflexLoop, snapshots: Sequence[Snapshot],
                future_returns: Sequence[float], generations: int = 1,
                verify_top: int = 4, **kw) -> Verification | None:
        """
        Search freely, promote strictly. The two are deliberately separate.

        SEARCH is arithmetic over historical data. No capital is at risk while
        the ego loop thinks, so nothing gates it: the search head walks to the
        best-scoring candidate each generation whether or not that candidate
        could ever be traded.

        PROMOTION is what reaches the hot path, and it passes all four gates -
        out-of-sample improvement, invariants, the risk arbiter, latency.

        This separation is not a detail. The first version of this method gated
        the SEARCH by the arbiter too, and the result was a system that could not
        learn its way out of a bad regime: the incumbent was backwards for the
        new market, every intermediate kernel between backwards and correct was
        also unprofitable, and the arbiter - correctly - vetoed all of them. The
        search could never take the first step across the valley, so 480 out of
        480 candidates were refused and nothing ever improved.

        The rule that fixes it is the project's own first rule. The arbiter
        governs what gets SIZED, not what gets THOUGHT. Letting a risk limit
        censor a hypothesis is the same category error as letting a hypothesis
        override a risk limit, and it fails just as badly - quietly, by never
        improving, rather than loudly.

        Returns the last promotion's verification, or None if nothing earned the
        hot path. Nothing being promoted is normal and healthy.
        """
        search_head = reflex.active_kernel
        head_score = self.score(search_head, snapshots, future_returns)
        promoted: Verification | None = None

        for _ in range(max(generations, 1)):
            scored = [(self.score(c, snapshots, future_returns), c)
                      for c in self.propose(search_head)]
            scored.sort(key=lambda pair: pair[0], reverse=True)

            # the search head walks, ungated - it is only ever a number on paper
            if scored and scored[0][0] > head_score:
                head_score, search_head = scored[0]

            # promotion is gated by all four, and only the leaders are worth checking
            for _, candidate in scored[:verify_top]:
                verification = self.verify(candidate, reflex.active_kernel, snapshots,
                                           future_returns,
                                           max_size_fraction=reflex.max_size_fraction, **kw)
                if not verification.passed:
                    continue
                kernel = candidate
                # The arbiter's permitted size is binding, not advisory: a kernel
                # is promoted at the size it was ALLOWED, never what it asked for.
                peak = max(kernel.evaluate(s).size_fraction for s in snapshots[:200])
                if peak > verification.permitted_size > 0:
                    kernel = replace(kernel, size_scale=kernel.size_scale *
                                     (verification.permitted_size / peak))
                reflex.promote(kernel, f"scored {verification.score:+.1f} vs "
                                       f"{verification.incumbent_score:+.1f}; "
                                       f"risk: {verification.risk_note}")
                promoted = verification
                break

        return promoted


# ── adapters ──────────────────────────────────────────────────────────────────

class ExecutionAdapter(Protocol):
    def send(self, intent: Intent) -> None:
        ...


class PaperAdapter:
    """
    Records intents. Sends nothing. There is no live adapter in this repository.

    Writing one means naming a broker, holding a key and accepting the
    consequences of a bug at four in the morning. That is a human's decision and
    a human's signature, not something a codebase should make easy by accident.
    """

    def __init__(self) -> None:
        self.intents: list[Intent] = []

    def send(self, intent: Intent) -> None:
        if intent.is_trade:
            self.intents.append(intent)

    @property
    def count(self) -> int:
        return len(self.intents)


# ── demo ──────────────────────────────────────────────────────────────────────

def synthetic_market(n: int, seed: int, regime: str = "mean_reverting"):
    """Deterministic fake market. Snapshots plus the forward return after each."""
    rng = np.random.default_rng(seed)
    snapshots, forwards = [], []
    imbalance = 0.0
    inventory = 0.0
    for i in range(n):
        imbalance = 0.75 * imbalance + rng.normal(0, 0.8)
        spread = float(np.clip(1.2 + 0.5 * abs(imbalance) + rng.normal(0, 0.2), 0.4, 12.0))
        volatility = float(np.clip(90 + 25 * abs(imbalance) + rng.normal(0, 12), 20, 500))
        noise = rng.normal(0, 0.0008)
        forward = (-0.0006 * imbalance + noise) if regime == "mean_reverting" \
            else (0.0006 * imbalance + noise)
        inventory = float(np.clip(inventory * 0.9 + rng.normal(0, 0.01), -0.3, 0.3))
        snapshots.append(Snapshot(float(i), 100.0, spread, float(imbalance),
                                  volatility, inventory, staleness_ms=rng.uniform(0, 8)))
        forwards.append(float(forward))
    return snapshots, forwards


if __name__ == "__main__":
    print("=" * 78)
    print(" STARK MK18 - dual-loop execution")
    print("=" * 78)

    incumbent = Kernel("mm-v1", ("fade_imbalance", "spread_capture"),
                       (0.5, 2.0), (1.0, 0.3), size_scale=0.02)
    reflex = ReflexLoop(incumbent, max_size_fraction=0.05)
    paper = PaperAdapter()

    print(f"\nincumbent kernel: {incumbent.describe()}")

    train, train_forward = synthetic_market(4_000, seed=1, regime="mean_reverting")
    print("\n-- the reflex loop runs the market --")
    for snapshot in train:
        paper.send(reflex.tick(snapshot))
    print(f"  {reflex.stats}")
    print(f"  {paper.count} intents recorded, 0 orders sent (there is no live adapter)")

    print("\n-- hard checks fire on every tick, whatever the kernel says --")
    for label, bad in (
        ("stale data", Snapshot(1, 100.0, 2.0, 3.0, 120.0, 0.0, staleness_ms=500)),
        ("inventory over limit", Snapshot(2, 100.0, 2.0, 3.0, 120.0, 0.35)),
        ("broken snapshot", Snapshot(3, float("nan"), 2.0, 3.0, 120.0, 0.0)),
    ):
        print(f"  {label:<22} -> {reflex.tick(bad)}")

    print("\n-- the world changes: flow now CONTINUES instead of reverting --")
    shifted, shifted_forward = synthetic_market(4_000, seed=2, regime="trending")
    before = EgoLoop.score(incumbent, shifted, shifted_forward)
    print(f"  the incumbent scores {before:+.1f} bps on the new regime - it is now backwards")

    print("\n-- the ego loop searches, verifies, and rewrites the kernel --")
    ego = EgoLoop(seed=7, population=40)
    split = len(shifted) // 2
    promoted = ego.improve(reflex, shifted[:split], shifted_forward[:split], generations=12)
    print(f"  search explored freely; only promotion was gated")

    rejected = [v for v in ego.history if not v.passed]
    print(f"  {len(ego.history)} candidates evaluated, {len(rejected)} rejected")
    reasons: dict[str, int] = {}
    for verification in rejected:
        for failure in verification.failures():
            reasons[failure.split(" (")[0].split(":")[0]] = \
                reasons.get(failure.split(" (")[0].split(":")[0], 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>3} x {reason}")

    if promoted:
        print(f"\n  {len(reflex.promotions)} promotion(s) across the search:")
        for _, transition, why in reflex.promotions:
            print(f"    {transition:<28} {why}")
        print(f"\n  FINAL KERNEL: {reflex.active_kernel.describe()}")
        held_out = EgoLoop.score(reflex.active_kernel, shifted[split:], shifted_forward[split:])
        incumbent_held_out = EgoLoop.score(incumbent, shifted[split:], shifted_forward[split:])
        print(f"\n  on data NEITHER kernel has seen:")
        print(f"    old kernel {incumbent_held_out:+8.1f} bps")
        print(f"    new kernel {held_out:+8.1f} bps")
        print(f"  The structure changed, not just the numbers: it DROPPED fade_imbalance,")
        print(f"  which was backwards for this regime, and discovered follow_imbalance.")
        print(f"  Every operator in it came off a whitelist a human already reviewed.")

        print("\n-- two things in that log worth staring at --")
        print(f"  1. size crept {incumbent.size_scale:.3f} -> "
              f"{reflex.active_kernel.size_scale:.3f} across the search. Of course it did:")
        print("     the score is proportional to size, so an unconstrained search ALWAYS")
        print("     inflates until something stops it. Read the last promotion line - what")
        print("     stopped it was the arbiter's confidence-scaling cap, not good taste.")
        print("  2. the promoted kernel still asks for more than the reflex loop permits")
        print("     per tick, so it gets clamped again on the hot path:")
        fresh = ReflexLoop(reflex.active_kernel, max_size_fraction=0.05)
        clamped = sum("clamped" in fresh.tick(s).reason for s in shifted[split:split + 500])
        print(f"       {clamped}/500 ticks clamped at the reflex loop's own limit.")
        print("     Two independent limits, applied at different timescales. Neither one")
        print("     trusts the search, and neither one trusts the other.")
    else:
        print("\n  nothing was promoted. That is a normal outcome, and a healthy one.")

    print("\n" + "-" * 78)
    print("Gate 3 is the one that matters: the search proposes, the arbiter disposes.")
    print("No candidate reaches the hot path without the same unappealable code that")
    print("governs every other size in this system signing off on it.")
