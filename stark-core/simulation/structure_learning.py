#!/usr/bin/env python3
"""
Level 2 - Causal structure discovery. Learning the graph instead of asserting it.

Phase 2 fitted parameters on edges a human drew. This module draws the edges.

The honest version of that sentence is longer, and the length is the point:

  Observational data cannot fully identify a causal graph. X -> Y and Y -> X
  imply exactly the same joint distribution, so no amount of watching will tell
  them apart. What observation CAN identify is the Markov equivalence class -
  the set of graphs that are indistinguishable from outside. Some edges come out
  oriented (a collider X -> Y <- Z leaves a statistical fingerprint that nothing
  else does); the rest come out as "adjacent, direction unknown", and this
  module says so rather than picking one and sounding confident.

  What breaks the tie is intervention. Force X and watch Y move while forcing Y
  leaves X alone, and the arrow is settled. STARK can do that - the SCM in
  causal_engine.py has do() - so `orient_with_interventions` closes the loop:
  discover what observation supports, then pay for the rest with experiments.

Method:
  1. skeleton      PC-style: drop an edge when the pair is conditionally
                   independent given some subset of their neighbours
                   (partial correlation, Fisher z-test)
  2. colliders     orient v-structures using the separating sets from step 1
  3. propagation   Meek rules R1-R3: orient whatever is forced by not creating
                   a new collider and not creating a cycle
  4. interventions optional, and the only step that can finish the job

An undirected edge here is not a failure. It is a correctly reported limit of
what the data can support, and it names the experiment that would resolve it.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import stats

if __package__ in (None, ""):  # run as a script: put the stark-core/ root on the path
    import pathlib as _pl, sys as _sys
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

from simulation.causal_engine import CausalGraph

# Interventional data generator: (variable, value, n, seed) -> {name: samples}
Intervener = Callable[[str, float, int, int], Mapping[str, np.ndarray]]


# ── conditional independence ──────────────────────────────────────────────────

def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None) -> float:
    """
    Correlation of x and y with the linear influence of z removed.

    Residualise both on z, then correlate what is left. If z explains the whole
    relationship, the residuals are unrelated and the correlation collapses -
    which is exactly the signature of a spurious edge through a common cause.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if z is None or (hasattr(z, "shape") and z.size == 0):
        sd = x.std() * y.std()
        return 0.0 if sd == 0 else float(np.corrcoef(x, y)[0, 1])
    z = np.atleast_2d(np.asarray(z, dtype=float))
    if z.shape[0] != len(x):
        z = z.T
    design = np.column_stack([np.ones(len(x)), z])
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    sd = rx.std() * ry.std()
    return 0.0 if sd < 1e-15 else float(np.corrcoef(rx, ry)[0, 1])


def independence_p_value(r: float, n: int, n_controls: int) -> float:
    """
    Fisher z-transform: under independence, z = atanh(r) sqrt(n - k - 3) ~ N(0,1).

    Returns the two-sided p-value for "these are independent". A LARGE p-value
    is evidence of independence, which is the direction this algorithm needs -
    and is also why it is fragile on small samples: failing to detect a
    dependence is not the same as establishing independence, and with n=50 you
    will fail to detect plenty. Feed it real sample sizes.
    """
    dof = n - n_controls - 3
    if dof <= 0:
        return 1.0
    r = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r) * np.sqrt(dof)
    return float(2.0 * (1.0 - stats.norm.cdf(abs(z))))


# ── the discovered object ─────────────────────────────────────────────────────

@dataclass
class DiscoveredGraph:
    """
    What the data supports: some oriented edges, some honestly undecided.

    `directed` holds (parent, child) pairs. `undirected` holds frozensets of
    adjacent pairs whose direction the evidence does not fix.
    """

    variables: list[str]
    directed: set[tuple[str, str]] = field(default_factory=set)
    undirected: set[frozenset[str]] = field(default_factory=set)
    separating_sets: dict[frozenset[str], tuple[str, ...]] = field(default_factory=dict)
    evidence: dict[tuple[str, str], str] = field(default_factory=dict)

    @property
    def n_edges(self) -> int:
        return len(self.directed) + len(self.undirected)

    @property
    def fully_oriented(self) -> bool:
        return not self.undirected

    def parents_of(self, node: str) -> tuple[str, ...]:
        return tuple(sorted(p for p, c in self.directed if c == node))

    def adjacent(self, node: str) -> set[str]:
        out = {p for p, c in self.directed if c == node} | {c for p, c in self.directed if p == node}
        return out | {v for e in self.undirected if node in e for v in e if v != node}

    def has_edge(self, a: str, b: str) -> bool:
        return (a, b) in self.directed or (b, a) in self.directed or frozenset((a, b)) in self.undirected

    def open_questions(self) -> list[str]:
        """The experiments that would finish the job. One line each, in plain English."""
        return [
            f"Is it {a} -> {b} or {b} -> {a}? Intervene on {a}: if {b} moves, the arrow points at {b}."
            for a, b in (tuple(sorted(e)) for e in self.undirected)
        ]

    def to_causal_graph(self, data: Mapping[str, Sequence[float]]) -> CausalGraph:
        """
        Build a runnable SCM from the discovered structure, with linear
        coefficients fitted by least squares.

        Only fully oriented graphs may be built. A graph with an undecided edge
        has no defined parent set, and quietly picking a direction to make the
        code run is how a guess becomes a number that somebody trades on.
        """
        if not self.fully_oriented:
            raise ValueError(
                f"{len(self.undirected)} edge(s) still undirected: "
                f"{[tuple(sorted(e)) for e in self.undirected]}. "
                "Orient them with interventions before building an SCM."
            )
        arrays = {k: np.asarray(v, dtype=float) for k, v in data.items()}
        graph = CausalGraph()
        for node in self._topological_order():
            parents = self.parents_of(node)
            if not parents:
                mean, sd = float(arrays[node].mean()), float(arrays[node].std())
                graph.add(node, (), lambda v, e, m=mean, s=sd: m + s * e)
                continue
            design = np.column_stack([np.ones(len(arrays[node]))] + [arrays[p] for p in parents])
            coef, *_ = np.linalg.lstsq(design, arrays[node], rcond=None)
            resid_sd = float(np.std(arrays[node] - design @ coef))

            def assignment(v, e, parents=parents, coef=coef, resid_sd=resid_sd):
                out = np.full_like(e, coef[0], dtype=float)
                for w, p in zip(coef[1:], parents):
                    out = out + w * v[p]
                return out + resid_sd * e

            graph.add(node, parents, assignment)
        return graph

    def _topological_order(self) -> list[str]:
        remaining = list(self.variables)
        order: list[str] = []
        while remaining:
            free = [n for n in remaining if all(p in order for p in self.parents_of(n))]
            if not free:
                raise ValueError("discovered graph contains a cycle")
            order.extend(free)
            remaining = [n for n in remaining if n not in order]
        return order

    def __str__(self) -> str:
        lines = [f"{len(self.variables)} variables, {self.n_edges} edges "
                 f"({len(self.directed)} oriented, {len(self.undirected)} undecided)"]
        for parent, child in sorted(self.directed):
            lines.append(f"  {parent:>12} -> {child:<12} {self.evidence.get((parent, child), '')}")
        for edge in sorted(tuple(sorted(e)) for e in self.undirected):
            lines.append(f"  {edge[0]:>12} -- {edge[1]:<12} direction not identified by this data")
        return "\n".join(lines)


# ── discovery ─────────────────────────────────────────────────────────────────

def discover_structure(
    data: Mapping[str, Sequence[float]],
    alpha: float = 0.01,
    max_conditioning_set: int = 3,
) -> DiscoveredGraph:
    """
    PC algorithm: start fully connected, delete edges that conditional
    independence rules out, then orient what the data can orient.

    `alpha` is the significance level for the independence test. Lower alpha
    deletes fewer edges (a conservative, denser graph); higher deletes more.
    There is no setting that makes this decision for you - it is a judgement
    about which error you would rather make.
    """
    names = list(data)
    arrays = {k: np.asarray(v, dtype=float) for k, v in data.items()}
    n = len(arrays[names[0]])
    for k, v in arrays.items():
        if len(v) != n:
            raise ValueError(f"'{k}' has {len(v)} rows; expected {n}")
    if n < 30:
        raise ValueError(f"structure discovery on {n} rows is guesswork; bring at least 30")

    adjacency = {frozenset(pair) for pair in itertools.combinations(names, 2)}
    sepsets: dict[frozenset[str], tuple[str, ...]] = {}

    # 1. skeleton
    for order in range(max_conditioning_set + 1):
        for pair in list(adjacency):
            a, b = tuple(pair)
            neighbours = [v for v in names if v not in pair and frozenset((a, v)) in adjacency]
            if len(neighbours) < order:
                continue
            for conditioning in itertools.combinations(sorted(neighbours), order):
                z = np.column_stack([arrays[c] for c in conditioning]) if conditioning else None
                r = partial_correlation(arrays[a], arrays[b], z)
                if independence_p_value(r, n, len(conditioning)) > alpha:
                    adjacency.discard(pair)
                    sepsets[pair] = conditioning
                    break

    graph = DiscoveredGraph(variables=names, undirected=set(adjacency), separating_sets=sepsets)

    # 2. v-structures: a -> c <- b, identifiable because a collider makes two
    #    independent causes DEPENDENT once you condition on their shared effect
    for c in names:
        for a, b in itertools.combinations(sorted(graph.adjacent(c)), 2):
            if graph.has_edge(a, b):
                continue                                   # a and b adjacent: not a v-structure
            sep = sepsets.get(frozenset((a, b)))
            if sep is not None and c not in sep:
                _orient(graph, a, c, "collider")
                _orient(graph, b, c, "collider")

    _apply_meek_rules(graph)
    return graph


def _orient(graph: DiscoveredGraph, parent: str, child: str, why: str) -> bool:
    edge = frozenset((parent, child))
    if edge not in graph.undirected:
        return False
    graph.undirected.discard(edge)
    graph.directed.add((parent, child))
    graph.evidence[(parent, child)] = f"({why})"
    return True


def _creates_cycle(graph: DiscoveredGraph, parent: str, child: str) -> bool:
    """Would parent -> child close a directed loop? Depth-first from child."""
    stack, seen = [child], set()
    while stack:
        node = stack.pop()
        if node == parent:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(c for p, c in graph.directed if p == node)
    return False


def _apply_meek_rules(graph: DiscoveredGraph) -> None:
    """
    Propagate orientations that are forced.

    R1: a -> b -- c with a, c non-adjacent  =>  b -> c
        (b -> c is forced: c -> b would be a collider we would already have found)
    R2: a -> b -> c and a -- c              =>  a -> c   (c -> a would be a cycle)
    R3: a -- b, a -- c, a -- d, b -> d, c -> d, b and c non-adjacent  =>  a -> d
    """
    changed = True
    while changed:
        changed = False
        for edge in list(graph.undirected):
            b, c = tuple(edge)
            for x, y in ((b, c), (c, b)):
                # R1
                if any(p != y and not graph.has_edge(p, y) for p, ch in graph.directed if ch == x):
                    changed |= _orient(graph, x, y, "Meek R1")
                    break
                # R2
                if any(m for m in graph.variables
                       if (x, m) in graph.directed and (m, y) in graph.directed):
                    changed |= _orient(graph, x, y, "Meek R2")
                    break
                # R3
                shared = [m for m in graph.variables
                          if frozenset((x, m)) in graph.undirected and (m, y) in graph.directed]
                if any(not graph.has_edge(m1, m2) for m1, m2 in itertools.combinations(shared, 2)):
                    changed |= _orient(graph, x, y, "Meek R3")
                    break


# ── interventions: the only thing that settles the rest ───────────────────────

def orient_with_interventions(
    graph: DiscoveredGraph,
    intervener: Intervener,
    n: int = 2_000,
    seed: int = 0,
    strength: float = 3.0,
    alpha: float = 0.01,
) -> DiscoveredGraph:
    """
    Settle undirected edges by experiment.

    For each undecided pair {a, b}: force a, and see whether b's distribution
    moves. Force b, and see whether a's moves. Movement in exactly one direction
    orients the edge. Movement in both means an unmeasured common cause or a
    feedback loop, and the edge stays undirected - which is the correct answer,
    not a shortfall.

    This is why STARK simulates. An intervention in the world costs money and
    may be impossible; an intervention in the model costs a random seed.
    """
    for edge in list(graph.undirected):
        a, b = tuple(sorted(edge))
        a_moves_b = _intervention_moves(intervener, cause=a, effect=b, n=n, seed=seed,
                                        strength=strength, alpha=alpha)
        b_moves_a = _intervention_moves(intervener, cause=b, effect=a, n=n, seed=seed + 1,
                                        strength=strength, alpha=alpha)
        if a_moves_b and not b_moves_a:
            _orient(graph, a, b, "intervention")
        elif b_moves_a and not a_moves_b:
            _orient(graph, b, a, "intervention")
        # both or neither: leave it undirected and let open_questions() report it
    _apply_meek_rules(graph)
    return graph


def _intervention_moves(
    intervener: Intervener, cause: str, effect: str,
    n: int, seed: int, strength: float, alpha: float,
) -> bool:
    """Does do(cause = high) shift `effect` compared with do(cause = low)?"""
    low = np.asarray(intervener(cause, -strength, n, seed)[effect], dtype=float)
    high = np.asarray(intervener(cause, +strength, n, seed)[effect], dtype=float)
    return float(stats.ttest_ind(low, high, equal_var=False).pvalue) < alpha


if __name__ == "__main__":
    print("STARK MK18 - causal structure discovery\n")

    # ── case 1: a collider. Observation alone is enough. ─────────────────────
    rng = np.random.default_rng(7)
    n = 4_000
    liquidity = rng.normal(0, 1, n)
    news = rng.normal(0, 1, n)
    flow = 0.8 * liquidity + 0.9 * news + rng.normal(0, 0.4, n)
    price = 1.2 * flow + rng.normal(0, 0.5, n)

    print("CASE 1 - truth: liquidity -> flow <- news, flow -> price")
    print("A collider leaves a fingerprint nothing else does: two independent causes")
    print("become DEPENDENT once you condition on their shared effect.\n")
    found = discover_structure(
        {"liquidity": liquidity, "news": news, "flow": flow, "price": price}, alpha=0.01)
    print(found)
    print(f"\nfully oriented from observation alone: {found.fully_oriented}")

    # ── case 2: a chain. Observation is NOT enough, and says so. ─────────────
    print("\n" + "=" * 74)
    supply = rng.normal(0, 1, n)
    inventory = 0.9 * supply + rng.normal(0, 0.4, n)
    spot = 1.1 * inventory + rng.normal(0, 0.4, n)
    chain_data = {"supply": supply, "inventory": inventory, "spot": spot}

    print("\nCASE 2 - truth: supply -> inventory -> spot")
    print("No collider anywhere. supply->inventory->spot, spot->inventory->supply and")
    print("inventory->both imply the SAME joint distribution. Watching cannot separate")
    print("them, and the honest output is an undirected edge.\n")
    chain = discover_structure(chain_data, alpha=0.01)
    print(chain)
    print("\nwhat the data cannot answer:")
    for question in chain.open_questions():
        print(f"  {question}")

    try:
        chain.to_causal_graph(chain_data)
    except ValueError as e:
        print(f"\nand it refuses to pretend otherwise:\n  {e}")

    # ── the experiments ──────────────────────────────────────────────────────
    truth = (CausalGraph()
             .add("supply", (), lambda v, e: e)
             .add("inventory", ("supply",), lambda v, e: 0.9 * v["supply"] + 0.4 * e)
             .add("spot", ("inventory",), lambda v, e: 1.1 * v["inventory"] + 0.4 * e))

    def intervene(variable, value, n, seed):
        return truth.do(n=n, seed=seed, **{variable: value}).values

    print("\n" + "=" * 74)
    print("\nSo run the experiments. Forcing supply moves inventory; forcing inventory")
    print("leaves supply alone. That asymmetry is the arrow.\n")
    orient_with_interventions(chain, intervene, n=2_000, seed=3)
    print(chain)

    print(f"\nfully oriented: {chain.fully_oriented}")
    if chain.fully_oriented:
        scm = chain.to_causal_graph(chain_data)
        base = scm.sample(n=2_000, seed=1)
        forced = scm.do(n=2_000, seed=1, inventory=3.0)
        print(f"\nthe learned SCM runs: do(inventory=3.0) moves spot "
              f"{base.mean('spot'):+.2f} -> {forced.mean('spot'):+.2f}")
        print("Structure discovered, coefficients fitted, counterfactuals answerable -")
        print("and every arrow either earned by a collider or paid for with an experiment.")
