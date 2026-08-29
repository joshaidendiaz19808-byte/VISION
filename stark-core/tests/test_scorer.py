#!/usr/bin/env python3
"""
Tests for the learned scorer.

Three groups. The machinery (does the boosting work), the honesty (is it
calibrated, does it peek at the future), and the one that matters most:

  ADVERSARIAL - a scorer deliberately corrupted to be confident and wrong is
  plugged into the real pipeline, and the red team and the arbiter must still
  refuse everything it recommends.

That last group is CLAUDE.md rule 1 written as an experiment. Until now the
arbiter had only ever had to hold against a rank-IC heuristic that was easy to
refuse. This is what it looks like when something persuasive argues for capital.

Run: pytest stark-core/tests/test_scorer.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from core.generator import Generator, rank_ic, shrink
from core.red_team import RedTeam
from core.risk_arbiter import Proposal, RiskArbiter, RiskVeto
from core.scorer import (PROFILE_FEATURES, Calibration, GradientBoostedTrees,
                         LearnedScorer, RegressionTree, out_of_sample_label, profile,
                         synthetic_corpus)


@pytest.fixture(scope="module")
def corpus():
    return synthetic_corpus(n_examples=700, length=300, seed=1)


@pytest.fixture(scope="module")
def fitted(corpus) -> LearnedScorer:
    scorer = LearnedScorer(GradientBoostedTrees(n_trees=50, learning_rate=0.1, seed=3))
    scorer.fit(*corpus)
    return scorer


def signal_and_noise(n=600, strength=0.25, seed=7):
    rng = np.random.default_rng(seed)
    real = rng.normal(0, 1, n)
    fwd = (strength * real + rng.normal(0, 1, n)) * 0.01
    return {"real_signal": real, "pure_noise": rng.normal(0, 1, n)}, fwd


# ── the machinery ─────────────────────────────────────────────────────────────

class TestTrees:
    def test_a_tree_learns_a_step_function(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, (600, 2))
        y = np.where(X[:, 0] > 0, 1.0, -1.0)
        tree = RegressionTree(max_depth=2, min_samples_leaf=20).fit(X, y)
        assert np.mean(np.sign(tree.predict(X)) == y) > 0.95

    def test_a_tree_on_pure_noise_predicts_the_mean(self):
        rng = np.random.default_rng(1)
        X, y = rng.normal(0, 1, (400, 3)), rng.normal(5.0, 0.01, 400)
        predictions = RegressionTree(max_depth=3, min_samples_leaf=40).fit(X, y).predict(X)
        assert np.std(predictions) < 0.02

    def test_boosting_reduces_error_against_a_single_tree(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(-2, 2, (800, 3))
        y = np.sin(X[:, 0]) + 0.3 * X[:, 1] ** 2
        one = RegressionTree(max_depth=3).fit(X, y).predict(X)
        many = GradientBoostedTrees(n_trees=60, seed=1).fit(X, y).predict(X)
        assert np.mean((many - y) ** 2) < np.mean((one - y) ** 2)

    def test_boosting_is_deterministic(self):
        rng = np.random.default_rng(3)
        X, y = rng.normal(0, 1, (300, 4)), rng.normal(0, 1, 300)
        a = GradientBoostedTrees(n_trees=20, seed=5).fit(X, y).predict(X)
        b = GradientBoostedTrees(n_trees=20, seed=5).fit(X, y).predict(X)
        assert np.array_equal(a, b)

    def test_importances_find_the_feature_that_matters(self):
        rng = np.random.default_rng(4)
        X = rng.normal(0, 1, (600, 4))
        y = 3.0 * X[:, 2] + 0.01 * rng.normal(0, 1, 600)
        model = GradientBoostedTrees(n_trees=40, seed=1).fit(X, y)
        assert int(np.argmax(model.importances)) == 2

    def test_too_little_data_is_refused(self):
        rng = np.random.default_rng(5)
        with pytest.raises(ValueError, match="not enough"):
            GradientBoostedTrees().fit(rng.normal(0, 1, (10, 3)), rng.normal(0, 1, 10))

    def test_mismatched_shapes_are_refused(self):
        rng = np.random.default_rng(6)
        with pytest.raises(ValueError):
            GradientBoostedTrees().fit(rng.normal(0, 1, (100, 3)), rng.normal(0, 1, 90))


# ── the profile: does it peek? ────────────────────────────────────────────────

class TestProfile:
    def test_the_profile_cannot_see_the_future(self):
        """
        The single most important property here. The profile is computed on the
        in-sample portion; the label comes from the rest. If changing the future
        changes the profile, the model is being trained on the answer.
        """
        rng = np.random.default_rng(1)
        n = 400
        x = rng.normal(0, 1, n)
        y = 0.2 * x + rng.normal(0, 1, n)
        original = profile(x, y, split=0.7)

        tampered = y.copy()
        tampered[int(n * 0.7):] = rng.normal(0, 50, n - int(n * 0.7))   # wreck the future
        assert np.allclose(original, profile(x, tampered, split=0.7)), "the profile peeked"

    def test_the_label_does_come_from_the_future(self):
        rng = np.random.default_rng(2)
        n = 400
        x = rng.normal(0, 1, n)
        y = 0.2 * x + rng.normal(0, 1, n)
        tampered = y.copy()
        tampered[int(n * 0.7):] = rng.normal(0, 1, n - int(n * 0.7))
        assert out_of_sample_label(x, y) != out_of_sample_label(x, tampered)

    def test_it_has_one_entry_per_named_meta_feature(self):
        rng = np.random.default_rng(3)
        assert len(profile(rng.normal(0, 1, 300), rng.normal(0, 1, 300))) == len(PROFILE_FEATURES)

    def test_a_decaying_edge_scores_low_on_decay(self):
        rng = np.random.default_rng(4)
        n = 600
        x = rng.normal(0, 1, n)
        strong_then_gone = np.linspace(0.6, 0.0, n) * x + rng.normal(0, 1, n)
        steady = 0.3 * x + rng.normal(0, 1, n)
        decay_index = PROFILE_FEATURES.index("ic_decay")
        assert profile(x, strong_then_gone)[decay_index] < profile(x, steady)[decay_index]

    def test_an_outlier_driven_correlation_shows_high_influence(self):
        rng = np.random.default_rng(5)
        n = 400
        clean = rng.normal(0, 1, n)
        spiky = clean.copy()
        spiky[rng.choice(n, 5, replace=False)] *= 20.0
        y = rng.normal(0, 1, n)
        index = PROFILE_FEATURES.index("max_influence")
        assert profile(spiky, 0.02 * spiky + y)[index] > profile(clean, 0.02 * clean + y)[index]

    def test_a_short_series_returns_zeros_rather_than_nonsense(self):
        assert np.all(profile([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0)


# ── honesty of the fitted model ───────────────────────────────────────────────

class TestLearnedScorer:
    def test_it_predicts_out_of_sample_ic_better_than_in_sample_ic_does(self, fitted):
        """The claim the whole module rests on. Measured on features never seen."""
        Xv, yv = synthetic_corpus(n_examples=300, length=300, seed=99)
        learned = np.clip(fitted.model.predict(Xv), 0, 1)
        naive = Xv[:, PROFILE_FEATURES.index("abs_ic")]
        assert np.mean(np.abs(learned - yv)) < np.mean(np.abs(naive - yv))

    def test_it_knows_apparent_edge_overstates_real_edge(self, fitted):
        Xv, yv = synthetic_corpus(n_examples=300, length=300, seed=98)
        naive = Xv[:, PROFILE_FEATURES.index("abs_ic")]
        learned = np.clip(fitted.model.predict(Xv), 0, 1)
        assert learned.mean() < naive.mean(), "the model did not learn that edge decays"

    def test_it_is_calibrated_on_held_out_rows(self, fitted):
        assert fitted.calibration is not None
        assert fitted.calibration.holdout_rows > 50
        assert fitted.calibration.trustworthy, str(fitted.calibration)

    def test_calibration_uses_rows_the_model_never_trained_on(self, corpus):
        X, y = corpus
        scorer = LearnedScorer(GradientBoostedTrees(n_trees=20, seed=1))
        calibration = scorer.fit(X, y, holdout=0.25)
        assert scorer.trained_rows + calibration.holdout_rows == len(X)

    def test_the_output_is_always_a_prior_in_range(self, fitted):
        rng = np.random.default_rng(11)
        for _ in range(40):
            value = fitted("f", rng.normal(0, 1, 300), rng.normal(0, 1, 300) * 0.01)
            assert 0.0 <= value <= 1.0 and np.isfinite(value)

    def test_it_refuses_to_score_before_it_is_fitted(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            LearnedScorer()("f", np.zeros(300), np.zeros(300))

    def test_it_explains_what_it_keys_on(self, fitted):
        explanation = fitted.explain()
        assert explanation and all(name in PROFILE_FEATURES for name, _ in explanation)

    def test_the_same_seed_gives_the_same_scorer(self, corpus):
        X, y = corpus
        a = LearnedScorer(GradientBoostedTrees(n_trees=20, seed=7)); a.fit(X, y)
        b = LearnedScorer(GradientBoostedTrees(n_trees=20, seed=7)); b.fit(X, y)
        rng = np.random.default_rng(1)
        x, fwd = rng.normal(0, 1, 300), rng.normal(0, 1, 300) * 0.01
        assert a("f", x, fwd) == b("f", x, fwd)


# ── it satisfies the contract the generator expects ───────────────────────────

class TestContract:
    def test_it_drops_into_the_generator(self, fitted):
        features, fwd = signal_and_noise()
        ideas = Generator(seed=1, scorer=fitted, min_prior=0.0).generate(features, fwd)
        assert ideas and ideas[0].feature == "real_signal"

    def test_the_generator_takes_the_more_sceptical_of_the_two(self):
        """
        Rule 1 in one assertion. A model that is certain cannot raise a prior
        above what the sample size supports - it can only ever lower it.
        """
        features, fwd = signal_and_noise(n=200)
        certain = lambda name, values, forward: 1.0
        for hypothesis in Generator(seed=1, scorer=certain, min_prior=0.0).generate(features, fwd):
            ceiling = shrink(hypothesis.raw_ic, hypothesis.n_obs)
            assert hypothesis.prior <= ceiling + 1e-12, "a confident model bought extra prior"

    def test_a_pessimistic_model_lowers_the_prior(self):
        features, fwd = signal_and_noise()
        gloomy = lambda name, values, forward: 0.001
        for hypothesis in Generator(seed=1, scorer=gloomy, min_prior=0.0).generate(features, fwd):
            assert hypothesis.prior <= 0.001 + 1e-12


# ── the adversarial group: rule 1, as an experiment ───────────────────────────

class TestPersuasiveAndWrong:
    """
    A scorer that is confident, articulate and completely wrong. Everything
    downstream must refuse it anyway.
    """

    LIAR = staticmethod(lambda name, values, forward: 0.99)

    def test_a_confident_liar_gets_noise_to_the_top_of_the_queue(self):
        """First establish that the corruption WORKS - otherwise the rest is vacuous."""
        rng = np.random.default_rng(3)
        noise = {f"noise_{i}": rng.normal(0, 1, 600) for i in range(6)}
        fwd = rng.normal(0, 1, 600) * 0.01
        ideas = Generator(seed=1, scorer=self.LIAR, min_prior=0.0).generate(noise, fwd)
        assert ideas, "the liar did not even produce hypotheses; nothing is being tested"

    def test_the_red_team_kills_what_the_liar_promotes(self):
        rng = np.random.default_rng(3)
        noise = {f"noise_{i}": rng.normal(0, 1, 600) for i in range(6)}
        fwd = rng.normal(0, 1, 600) * 0.01
        team = RedTeam(seed=42, n_permutations=400)
        ideas = Generator(seed=1, scorer=self.LIAR, min_prior=0.0).generate(noise, fwd)
        survivors = [h for h in ideas if team.run(h, noise[h.feature], fwd).survived]
        assert not survivors, f"{len(survivors)} noise features survived a hostile red team"

    def test_the_arbiter_vetoes_the_liar_even_at_full_confidence(self):
        """
        The last line. Even if the red team were fooled, a confident model
        cannot buy size that the evidence does not support.
        """
        arbiter = RiskArbiter()
        ruling = arbiter.rule(Proposal(
            hypothesis_id="H-LIAR", size_fraction=0.20, confidence=0.99,
            expected_edge_bps=0.4,          # the truth: there is no edge here
            n_trades=5_000, survived_red_team=True, volatility_bps=120.0))
        assert ruling.vetoed
        with pytest.raises(RiskVeto):
            ruling.enforce()

    def test_an_overfitted_scorer_does_not_change_what_survives(self):
        """
        The real test of rule 1: swap the intuition layer for a corrupted one and
        the set of surviving hypotheses must not grow. Intuition proposes; it
        does not decide.
        """
        features, fwd = signal_and_noise(n=800, strength=0.2)
        team = RedTeam(seed=42, n_permutations=300)

        def survivors(scorer):
            ideas = Generator(seed=1, scorer=scorer, min_prior=0.0).generate(features, fwd)
            return {h.feature for h in ideas if team.run(h, features[h.feature], fwd).survived}

        honest = survivors(None)
        corrupted = survivors(self.LIAR)
        assert corrupted <= honest, (
            f"a corrupted scorer got {corrupted - honest} past the red team"
        )

    def test_confidence_still_only_ever_falls(self, fitted):
        """Whatever the model says, the red team may only reduce it."""
        features, fwd = signal_and_noise(n=800)
        team = RedTeam(seed=42, n_permutations=300)
        for hypothesis in Generator(seed=1, scorer=fitted, min_prior=0.0).generate(features, fwd):
            report = team.run(hypothesis, features[hypothesis.feature], fwd)
            assert report.adjusted_confidence(hypothesis.prior) <= hypothesis.prior + 1e-12

    def test_the_scorer_cannot_reach_the_arbiter_at_all(self):
        """
        Structural, not behavioural: risk_arbiter.py imports nothing from STARK,
        so there is no code path by which any model can influence a ruling.
        """
        import ast
        import pathlib

        tree = ast.parse((pathlib.Path(__file__).resolve().parents[1]
                          / "core" / "risk_arbiter.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "core" not in imported and "scorer" not in imported
