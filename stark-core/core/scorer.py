#!/usr/bin/env python3
"""
Level 1 - the learned scorer. The seam Phase 1 left open, closed.

`Generator.scorer` has always been a pluggable hook with a rank-IC heuristic
standing in it. This is the model that goes there.

WHAT IT PREDICTS, AND WHY THAT CHOICE MATTERS

The obvious thing to learn is "will this feature make money". That target is a
trap: it trains a model to find in-sample edge, which is the one thing that is
never in short supply.

So it learns something else. Given only what is visible IN SAMPLE - the apparent
IC, how stable it is across folds, how fat-tailed the feature is, whether one
observation is carrying the whole correlation - it predicts what the OUT OF
SAMPLE IC will be. It is a model of decay, not of edge. It has been shown
thousands of features that looked good and then died, and it has learned what
that looks like on the way in.

A model like that is useful precisely because its answer is usually "less than
you think".

WHAT IT IS, HONESTLY

Gradient-boosted regression trees, written in numpy, about two hundred lines.
Not a neural network, and it is not called one anywhere in this file. On tabular
data of this size and shape, boosted trees are the strong baseline and a small
MLP is not; using the fashionable option and losing to the boring one would be a
bad trade for this system.

WHERE IT SITS IN THE ARCHITECTURE

CLAUDE.md rule 1: models handle intuition, and are never trusted with execution
logic. This one emits a PRIOR in [0, 1] and nothing else. It cannot size, cannot
veto, cannot appeal. Everything it proposes still faces all six red-team attacks
and then the arbiter, and the generator takes the MORE SCEPTICAL of the model's
opinion and the shrunk IC - so a confident model can lower a prior and can never
raise it past what the sample size supports.

The test suite includes a deliberately corrupted scorer - one trained on leaked
labels, that returns near-1.0 confidence on pure noise - to check that the rest
of the system still refuses. That is rule 1 stated as an experiment rather than
an intention.

numpy only, deterministic given a seed, no new dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

if __package__ in (None, ""):
    import pathlib as _pl, sys as _sys
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

from core.generator import rank_ic, shrink

# What the model is shown. All of it is computable at decision time, in sample.
PROFILE_FEATURES = (
    "abs_ic",             # apparent edge
    "log_n",              # how much data it rests on
    "ic_tstat",           # edge relative to its own noise
    "fold_agreement",     # does the sign hold across folds
    "ic_decay",           # second half of in-sample vs first half
    "max_influence",      # how much of the IC one observation carries
    "feature_kurtosis",   # fat tails: IC driven by a handful of points
    "feature_autocorr",   # persistent feature or noise
    "unique_fraction",    # discreteness
    "return_autocorr",    # the regime the feature is being judged in
)


# ── the meta-features ─────────────────────────────────────────────────────────

def profile(values: Sequence[float], forward_returns: Sequence[float],
            split: float = 0.7, folds: int = 5) -> np.ndarray:
    """
    Describe a feature using ONLY what is visible in sample.

    The split is the whole design. Everything here is computed on the first
    `split` of the data, because that is what a decision-maker can see; the label
    this model is trained against comes from the rest, which they cannot.
    """
    x = np.asarray(values, dtype=float)
    y = np.asarray(forward_returns, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 20:
        return np.zeros(len(PROFILE_FEATURES))

    cut = max(int(n * split), 10)
    xs, ys = x[:cut], y[:cut]
    ic = rank_ic(xs, ys)
    abs_ic = abs(ic)

    tstat = abs_ic * np.sqrt(max(cut - 2, 1) / max(1e-9, 1 - min(abs_ic ** 2, 0.999999)))

    fold_ics = [rank_ic(a, b) for a, b in
                zip(np.array_split(xs, folds), np.array_split(ys, folds))]
    dominant = np.sign(np.mean(fold_ics)) or 1.0
    agreement = float(np.mean([np.sign(v) == dominant for v in fold_ics]))

    half = cut // 2
    early, late = abs(rank_ic(xs[:half], ys[:half])), abs(rank_ic(xs[half:], ys[half:]))
    decay = float(late / early) if early > 1e-9 else 0.0

    # How much of the correlation rests on a single observation? Leave-one-out on
    # the most extreme points is cheap and catches the classic "one trade did it".
    influence = 0.0
    if cut > 30:
        extremes = np.argsort(np.abs(xs - xs.mean()))[-5:]
        for index in extremes:
            keep = np.ones(cut, dtype=bool)
            keep[index] = False
            influence = max(influence, abs(abs_ic - abs(rank_ic(xs[keep], ys[keep]))))

    centred = xs - xs.mean()
    sd = xs.std()
    kurtosis = float(np.mean(centred ** 4) / max(sd ** 4, 1e-12) - 3.0) if sd > 0 else 0.0

    return np.array([
        abs_ic,
        float(np.log1p(cut)),
        float(min(tstat, 20.0)),
        agreement,
        float(min(decay, 3.0)),
        float(influence),
        float(np.clip(kurtosis, -5.0, 50.0)),
        _autocorr(xs),
        float(len(np.unique(xs)) / cut),
        _autocorr(ys),
    ])


def _autocorr(v: np.ndarray) -> float:
    if len(v) < 3 or v.std() == 0:
        return 0.0
    return float(np.clip(np.corrcoef(v[:-1], v[1:])[0, 1], -1.0, 1.0))


def out_of_sample_label(values: Sequence[float], forward_returns: Sequence[float],
                        split: float = 0.7) -> float:
    """The truth the model is trained against: |IC| on the part it never saw."""
    x = np.asarray(values, dtype=float)
    y = np.asarray(forward_returns, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    cut = max(int(len(x) * split), 10)
    return float(abs(rank_ic(x[cut:], y[cut:]))) if len(x) - cut >= 10 else 0.0


# ── gradient-boosted regression trees, in numpy ───────────────────────────────

@dataclass
class _Node:
    feature: int = -1
    threshold: float = 0.0
    value: float = 0.0
    left: "_Node | None" = None
    right: "_Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


class RegressionTree:
    """
    A depth-limited CART regression tree. Greedy squared-error splits over
    quantile candidates - exhaustive search over every unique value buys
    accuracy this problem does not need and costs time it cannot spare.
    """

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 20,
                 candidates: int = 24) -> None:
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.candidates = int(candidates)
        self.root: _Node | None = None
        self.importances: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionTree":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.importances = np.zeros(X.shape[1])
        self.root = self._grow(X, y, depth=0)
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        node = _Node(value=float(y.mean()) if len(y) else 0.0)
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or y.std() == 0:
            return node

        best_gain, best_split = 0.0, None
        parent_sse = float(((y - y.mean()) ** 2).sum())
        for feature in range(X.shape[1]):
            column = X[:, feature]
            if column.std() == 0:
                continue
            thresholds = np.unique(np.quantile(column, np.linspace(0.05, 0.95,
                                                                  self.candidates)))
            for threshold in thresholds:
                mask = column <= threshold
                if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
                    continue
                left, right = y[mask], y[~mask]
                sse = float(((left - left.mean()) ** 2).sum() +
                            ((right - right.mean()) ** 2).sum())
                gain = parent_sse - sse
                if gain > best_gain:
                    best_gain, best_split = gain, (feature, float(threshold), mask)

        if best_split is None:
            return node
        feature, threshold, mask = best_split
        if self.importances is not None:
            self.importances[feature] += best_gain
        node.feature, node.threshold = feature, threshold
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.array([self._traverse(row) for row in X])

    def _traverse(self, row: np.ndarray) -> float:
        node = self.root
        while node is not None and not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value if node else 0.0


class GradientBoostedTrees:
    """
    Squared-loss gradient boosting. Deterministic given a seed - row subsampling
    is the only stochastic part and it draws from a seeded generator.
    """

    def __init__(self, n_trees: int = 60, learning_rate: float = 0.08,
                 max_depth: int = 3, min_samples_leaf: int = 20,
                 subsample: float = 0.8, seed: int = 0) -> None:
        self.n_trees = int(n_trees)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.seed = int(seed)
        self.base = 0.0
        self.trees: list[RegressionTree] = []
        self.importances: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedTrees":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(X) != len(y):
            raise ValueError(f"{len(X)} rows of X against {len(y)} labels")
        if len(X) < 50:
            raise ValueError(f"{len(X)} training rows is not enough to learn anything")

        rng = np.random.default_rng(self.seed)
        self.base = float(y.mean())
        prediction = np.full(len(y), self.base)
        self.trees = []
        self.importances = np.zeros(X.shape[1])
        take = max(int(len(X) * self.subsample), self.min_samples_leaf * 2)

        for _ in range(self.n_trees):
            residual = y - prediction
            rows = rng.choice(len(X), size=take, replace=False)
            tree = RegressionTree(self.max_depth, self.min_samples_leaf).fit(
                X[rows], residual[rows])
            prediction += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
            if tree.importances is not None:
                self.importances += tree.importances
        if self.importances.sum() > 0:
            self.importances /= self.importances.sum()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.full(len(X), self.base)
        for tree in self.trees:
            out += self.learning_rate * tree.predict(X)
        return out


# ── the scorer ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Calibration:
    """Does a predicted 0.6 actually mean 0.6? If not, the number is a decoration."""

    bins: tuple[tuple[float, float, int], ...]     # (mean predicted, mean actual, count)
    mean_absolute_error: float
    calibration_error: float
    holdout_rows: int

    @property
    def trustworthy(self) -> bool:
        return self.calibration_error < 0.05

    def __str__(self) -> str:
        head = (f"holdout {self.holdout_rows} rows | MAE {self.mean_absolute_error:.4f} "
                f"| calibration error {self.calibration_error:.4f}"
                + ("" if self.trustworthy else "  <- NOT CALIBRATED, treat as a ranking only"))
        rows = "\n".join(f"    predicted {p:.3f} -> actual {a:.3f}  (n={n})"
                         for p, a, n in self.bins)
        return head + ("\n" + rows if rows else "")


class LearnedScorer:
    """
    A fitted model of out-of-sample decay, wearing the `Scorer` contract.

    Call it exactly as the heuristic was called. It returns a prior in [0, 1] and
    has no other powers.
    """

    def __init__(self, model: GradientBoostedTrees | None = None, seed: int = 0,
                 split: float = 0.7) -> None:
        self.model = model or GradientBoostedTrees(seed=seed)
        self.split = float(split)
        self.calibration: Calibration | None = None
        self.trained_rows = 0

    def fit(self, X: np.ndarray, y: np.ndarray, holdout: float = 0.25) -> Calibration:
        """
        Fit, then measure calibration on rows the model never saw.

        The holdout is not optional. A scorer whose confidence is uncalibrated is
        worse than one with no confidence at all, because the number gets used.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        cut = int(len(X) * (1.0 - holdout))
        self.model.fit(X[:cut], y[:cut])
        self.trained_rows = cut

        predicted = np.clip(self.model.predict(X[cut:]), 0.0, 1.0)
        actual = y[cut:]
        self.calibration = _calibrate(predicted, actual)
        return self.calibration

    def __call__(self, name: str, values: np.ndarray, forward_returns: np.ndarray) -> float:
        """The Scorer contract. A prior in [0, 1], and nothing else."""
        if not self.model.trees:
            raise RuntimeError("scorer has not been fitted - call fit() first")
        prediction = float(self.model.predict(profile(values, forward_returns,
                                                      self.split).reshape(1, -1))[0])
        return float(np.clip(prediction, 0.0, 1.0))

    def explain(self, top: int = 5) -> list[tuple[str, float]]:
        """What the model actually keys on. Look at this before trusting it."""
        if self.model.importances is None:
            return []
        order = np.argsort(self.model.importances)[::-1][:top]
        return [(PROFILE_FEATURES[i], float(self.model.importances[i])) for i in order]


def _calibrate(predicted: np.ndarray, actual: np.ndarray, n_bins: int = 5) -> Calibration:
    bins: list[tuple[float, float, int]] = []
    edges = np.quantile(predicted, np.linspace(0, 1, n_bins + 1))
    errors = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (predicted >= low) & (predicted <= high)
        if mask.sum() < 5:
            continue
        mean_predicted, mean_actual = float(predicted[mask].mean()), float(actual[mask].mean())
        bins.append((mean_predicted, mean_actual, int(mask.sum())))
        errors.append(abs(mean_predicted - mean_actual))
    return Calibration(tuple(bins),
                       float(np.mean(np.abs(predicted - actual))) if len(predicted) else float("nan"),
                       float(np.mean(errors)) if errors else float("nan"),
                       int(len(predicted)))


# ── training data ─────────────────────────────────────────────────────────────

def synthetic_corpus(n_examples: int = 1_500, length: int = 400, seed: int = 0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a training set of features that survived and features that did not.

    Deliberately mixed: genuine stable edges, edges that decay, edges present in
    only one regime, correlations carried by a handful of outliers, and pure
    noise. The model's whole job is telling those apart from the in-sample side,
    so the corpus has to contain all of them in realistic proportions.

    This is SYNTHETIC. A scorer fitted here has learned the decay patterns of
    this generator, not of any market. Refit on real feature history before it
    means anything - the code takes any corpus, the honesty is in the label.
    """
    rng = np.random.default_rng(seed)
    profiles, labels = [], []

    for _ in range(n_examples):
        kind = rng.integers(0, 5)
        noise = rng.normal(0, 1, length)

        if kind == 0:                                   # a real, stable edge
            feature = rng.normal(0, 1, length)
            fwd = rng.uniform(0.05, 0.3) * feature + noise
        elif kind == 1:                                 # an edge that decays
            feature = rng.normal(0, 1, length)
            strength = np.linspace(rng.uniform(0.2, 0.4), 0.0, length)
            fwd = strength * feature + noise
        elif kind == 2:                                 # works in one regime only
            feature = rng.normal(0, 1, length)
            regime = np.arange(length) < length // 2
            fwd = np.where(regime, 0.35 * feature, -0.05 * feature) + noise
        elif kind == 3:                                 # carried by a few outliers
            feature = rng.normal(0, 1, length)
            feature[rng.choice(length, 6, replace=False)] *= 12.0
            fwd = 0.02 * feature + noise
        else:                                           # pure noise
            feature = rng.normal(0, 1, length)
            fwd = noise

        profiles.append(profile(feature, fwd * 0.01))
        labels.append(out_of_sample_label(feature, fwd * 0.01))

    return np.array(profiles), np.array(labels)


def heuristic_baseline(values: np.ndarray, forward_returns: np.ndarray) -> float:
    """The incumbent, isolated so the learned model can be measured against it."""
    return float(min(abs(rank_ic(np.asarray(values, float)[: int(len(values) * 0.7)],
                                 np.asarray(forward_returns, float)[: int(len(values) * 0.7)])),
                     1.0))


if __name__ == "__main__":
    print("STARK MK18 - the learned scorer\n")

    print("Building a corpus of features that survived and features that died...")
    X, y = synthetic_corpus(n_examples=1_500, length=400, seed=1)
    print(f"  {len(X)} examples, {X.shape[1]} meta-features each")
    print(f"  out-of-sample |IC| ranges {y.min():.3f} to {y.max():.3f}, mean {y.mean():.3f}\n")

    scorer = LearnedScorer(GradientBoostedTrees(n_trees=80, learning_rate=0.08, seed=3))
    calibration = scorer.fit(X, y)
    print("Fitted. Calibration on rows it never saw:")
    print(f"  {calibration}\n")

    print("What it keys on:")
    for name, weight in scorer.explain():
        print(f"  {weight:6.1%}  {name}")

    # honest comparison against the incumbent, on fresh data
    Xv, yv = synthetic_corpus(n_examples=600, length=400, seed=99)
    learned = np.clip(scorer.model.predict(Xv), 0, 1)
    baseline = Xv[:, PROFILE_FEATURES.index("abs_ic")]        # in-sample |IC|, the heuristic
    print(f"\nPredicting out-of-sample |IC| on 600 unseen features:")
    print(f"  in-sample |IC| as the predictor : MAE {np.mean(np.abs(baseline - yv)):.4f}")
    print(f"  learned scorer                  : MAE {np.mean(np.abs(learned - yv)):.4f}")
    improvement = 1 - np.mean(np.abs(learned - yv)) / np.mean(np.abs(baseline - yv))
    print(f"  {improvement:.0%} lower error - it has learned that apparent edge decays,")
    print(f"  and by roughly how much.")

    print("\nThe thing it learned, in one line:")
    print(f"  mean in-sample |IC| across the corpus : {baseline.mean():.3f}")
    print(f"  mean out-of-sample |IC|               : {yv.mean():.3f}")
    print(f"  mean learned prediction               : {learned.mean():.3f}")
    print("  It predicts the smaller number, because the smaller number is the true one.")

    # ── in the pipeline ──────────────────────────────────────────────────────
    from core.generator import Generator
    from core.red_team import RedTeam

    print("\n" + "=" * 74)
    print(" IN THE PIPELINE")
    print("=" * 74)

    rng = np.random.default_rng(7)
    n = 800
    real = rng.normal(0, 1, n)
    decaying = rng.normal(0, 1, n)
    features = {
        "order_flow_imbalance": real,
        "decaying_signal": decaying,
        "rsi_14": rng.normal(0, 1, n),
        "moon_phase": rng.normal(0, 1, n),
    }
    fwd = (0.18 * real + np.linspace(0.5, 0.0, n) * decaying + rng.normal(0, 1, n)) * 0.01

    team = RedTeam(seed=42, n_permutations=400)
    print("\nfeature              heuristic   learned   verdict")
    print("-" * 62)
    heuristic_ideas = {h.feature: h for h in
                       Generator(seed=1, min_prior=0.0).generate(features, fwd)}
    learned_ideas = {h.feature: h for h in
                     Generator(seed=1, scorer=scorer, min_prior=0.0).generate(features, fwd)}
    for name in features:
        h_prior = heuristic_ideas[name].prior if name in heuristic_ideas else 0.0
        l_prior = learned_ideas[name].prior if name in learned_ideas else 0.0
        hypothesis = learned_ideas.get(name) or heuristic_ideas.get(name)
        report = team.run(hypothesis, features[name], fwd)
        verdict = "survives" if report.survived else "killed by " + \
            report.kill_shots[0].name
        print(f"  {name:<20} {h_prior:6.3f}    {l_prior:6.3f}    {verdict}")

    print("\n  Look at decaying_signal: strong early, gone by the end. The heuristic")
    print("  rates it on the whole sample; the learned scorer has seen that shape")
    print("  thousands of times and marks it down before the red team even runs.")

    # ── and now the adversarial case ─────────────────────────────────────────
    print("\n" + "=" * 74)
    print(" WHAT IF THE MODEL IS CONFIDENT AND WRONG?")
    print("=" * 74)

    liar = lambda name, values, forward: 0.99
    noise_only = {f"noise_{i}": rng.normal(0, 1, n) for i in range(6)}
    noise_fwd = rng.normal(0, 1, n) * 0.01

    promoted = Generator(seed=1, scorer=liar, min_prior=0.0).generate(noise_only, noise_fwd)
    print(f"\nA scorer returning 0.99 on everything promotes {len(promoted)} pure-noise")
    print("features to the top of the queue. Then:")
    survivors = 0
    for hypothesis in promoted:
        report = team.run(hypothesis, noise_only[hypothesis.feature], noise_fwd)
        if report.survived:
            survivors += 1
        else:
            print(f"  {hypothesis.feature:<10} killed by "
                  f"{', '.join(a.name for a in report.kill_shots)}")
    print(f"\n  {survivors} of {len(promoted)} survived.")
    print("  The intuition layer proposes. It does not decide, and it never has.")
    print("  That is rule 1, and it is now tested against something that argues back.")
