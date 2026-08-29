#!/usr/bin/env python3
"""
Phase 2 tests: the continuous-time engine and the real-time inference loop.

A simulator is only worth what its assumptions are worth, so these tests check
the assumptions rather than the outputs:

  1. integrator   - does Euler-Maruyama reproduce known analytic moments?
  2. calibration  - can the fitter recover parameters we planted?
  3. mechanism    - do the models do the physical thing they claim (delay
                    delays, spreads widen with flow, impact is concave)?
  4. counterfactual - do branches differ by the ACTION and not by luck?
  5. real time    - does the filter actually beat the frozen model, and does it
                    know when it is wrong?

Run: pytest stark-core/tests/test_simulation.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.active_inference import (
    ActiveInferenceLoop,
    OnlineResult,
    PredictionErrorLoop,
    PredictiveProcess,
    compare_online_vs_static,
)
from simulation.causal_engine import (
    CausalGraph,
    Fit,
    ItoProcess,
    OrderBookMicrostructure,
    OrnsteinUhlenbeck,
    Path,
    SDEMultiverse,
    SupplyChainDelay,
    fit_impact,
    fit_ornstein_uhlenbeck,
)


# ── 1. the integrator ─────────────────────────────────────────────────────────

class TestIntegrator:
    def test_ou_reaches_its_analytic_stationary_distribution(self):
        """
        The one test that says the integrator is real: an OU process has a known
        stationary mean (mu) and sd (sigma / sqrt(2 theta)). Hit both or the
        scheme is wrong.
        """
        ou = OrnsteinUhlenbeck(theta=2.0, mu=5.0, sigma=1.0, x0=0.0)
        x = ou.integrate(n_steps=40_000, dt=0.01, seed=1)["x"][2_000:]  # drop burn-in
        assert np.mean(x) == pytest.approx(5.0, abs=0.05)
        assert np.std(x) == pytest.approx(ou.stationary_sd, rel=0.05)

    def test_half_life_is_the_time_to_decay_by_half(self):
        ou = OrnsteinUhlenbeck(theta=2.0, mu=0.0, sigma=1e-9, x0=1.0)   # near-noiseless
        path = ou.integrate(n_steps=2_000, dt=0.001, seed=1)
        at_half_life = float(np.interp(ou.half_life, path.times, path["x"]))
        assert at_half_life == pytest.approx(0.5, abs=0.01)

    def test_exact_transition_matches_fine_grained_simulation(self):
        """
        OU overrides predict() with the exact transition density. Check it against
        Monte Carlo built from MANY SMALL Euler steps - comparing it to one coarse
        step would only measure Euler's discretisation error, which is the thing
        the exact transition exists to avoid.
        """
        ou = OrnsteinUhlenbeck(theta=3.0, mu=1.0, sigma=0.7, x0=0.0)
        horizon, fine_dt = 0.25, 0.001
        n_steps = int(horizon / fine_dt)
        mean, var = ou.predict(np.array([0.0]), horizon)

        endpoints = np.array([ou.integrate(n_steps, fine_dt, seed=s)["x"][-1] for s in range(800)])
        stderr = float(np.std(endpoints) / np.sqrt(len(endpoints)))
        assert abs(float(np.mean(endpoints)) - float(mean[0])) < 4 * stderr
        assert float(np.var(endpoints)) == pytest.approx(float(var[0]), rel=0.15)

    def test_zero_diffusion_is_deterministic(self):
        ou = OrnsteinUhlenbeck(theta=1.0, mu=2.0, sigma=0.0, x0=0.0)
        a = ou.integrate(500, 0.01, seed=1)["x"]
        b = ou.integrate(500, 0.01, seed=999)["x"]
        assert np.allclose(a, b), "with no noise, the seed must not matter"

    def test_same_seed_same_path(self):
        book = OrderBookMicrostructure()
        assert np.allclose(book.integrate(300, 0.001, seed=7).states,
                           book.integrate(300, 0.001, seed=7).states)

    def test_different_seeds_differ(self):
        book = OrderBookMicrostructure()
        assert not np.allclose(book.integrate(300, 0.001, seed=7).states,
                               book.integrate(300, 0.001, seed=8).states)

    def test_correlated_noise_matches_the_requested_correlation(self):
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])

        class Pair(ItoProcess):
            names = ("a", "b")
            correlation = corr

            def drift(self, t, x, path, i):
                return np.zeros(2)

            def diffusion(self, t, x):
                return np.ones(2)

        dw = Pair().draw_increments(50_000, 0.01, seed=2)
        assert np.corrcoef(dw.T)[0, 1] == pytest.approx(0.8, abs=0.02)

    def test_wrong_shaped_increments_are_rejected(self):
        with pytest.raises(ValueError):
            OrnsteinUhlenbeck().integrate(10, 0.01, increments=np.zeros((10, 3)))


# ── 2. calibration ────────────────────────────────────────────────────────────

class TestCalibration:
    @pytest.mark.parametrize("theta, mu, sigma, dt", [
        (3.0, 1.5, 0.8, 0.005),      # theta*dt = 0.015
        (0.5, -2.0, 0.3, 0.01),      # theta*dt = 0.005
        (8.0, 0.0, 2.0, 0.002),      # theta*dt = 0.016
    ])
    def test_parameter_recovery(self, theta, mu, sigma, dt):
        """
        Plant known parameters, recover them. A fitter that cannot pass this has
        no business touching data whose answer you do not already know.

        Judged against the fit's OWN error bars rather than a fixed percentage.
        theta is the noisiest of the three (its sd here is ~0.07 on a truth of
        0.5), so a flat 15% tolerance would fail on an ordinary draw and pass on
        a lucky one - which tests the seed, not the fitter.

        dt is chosen per case to keep theta*dt <= 0.02, inside the range where
        Euler-Maruyama produces a faithful OU sample. See the next test for what
        happens outside it.
        """
        truth = OrnsteinUhlenbeck(theta=theta, mu=mu, sigma=sigma, x0=mu)
        series = truth.integrate(20_000, dt, seed=11)["x"]
        fit = fit_ornstein_uhlenbeck(series, dt=dt)
        assert fit.converged
        for name, planted in (("theta", theta), ("mu", mu), ("sigma", sigma)):
            err = fit.stderr[name]
            assert abs(fit.params[name] - planted) < 3 * err, (
                f"{name}: recovered {fit.params[name]:.4f} vs planted {planted} "
                f"({abs(fit.params[name] - planted) / err:.1f} sigma)"
            )

    def test_coarse_steps_bias_the_fit_and_finer_steps_fix_it(self):
        """
        Documents a real limitation rather than hiding it.

        Data generated by Euler-Maruyama at theta*dt = 0.08 and then fitted with
        the EXACT OU transition recovers sigma several standard errors high -
        the generator and the estimator genuinely disagree about the model at
        that step size. Halving dt to theta*dt = 0.016 removes it. Anyone
        integrating a fast process needs to know this before trusting a fit.
        """
        truth = OrnsteinUhlenbeck(theta=8.0, mu=0.0, sigma=2.0)
        coarse = fit_ornstein_uhlenbeck(truth.integrate(20_000, 0.01, seed=11)["x"], 0.01)
        fine = fit_ornstein_uhlenbeck(truth.integrate(20_000, 0.002, seed=11)["x"], 0.002)
        coarse_err = abs(coarse.params["sigma"] - 2.0) / coarse.stderr["sigma"]
        fine_err = abs(fine.params["sigma"] - 2.0) / fine.stderr["sigma"]
        assert coarse_err > 3.0, "the coarse-step bias has vanished - re-check this limit"
        assert fine_err < coarse_err / 2, "finer steps must reduce the discretisation bias"

    def test_true_values_sit_inside_the_error_bars(self):
        truth = OrnsteinUhlenbeck(theta=3.0, mu=1.5, sigma=0.8, x0=1.5)
        fit = fit_ornstein_uhlenbeck(truth.integrate(20_000, 0.01, seed=3)["x"], dt=0.01)
        for name, true_value in (("theta", 3.0), ("mu", 1.5), ("sigma", 0.8)):
            err = fit.stderr[name]
            assert np.isfinite(err) and err > 0, f"{name} has no usable error bar"
            assert abs(fit.params[name] - true_value) < 3 * err, f"{name} is >3 sigma from truth"

    def test_standard_errors_are_not_secretly_the_parameters(self):
        """
        Regression: L-BFGS-B's hess_inv can come back near-identity, which made
        every stderr equal its own parameter and looked like a real error bar.
        """
        fit = fit_ornstein_uhlenbeck(
            OrnsteinUhlenbeck(theta=3.0, mu=1.5, sigma=0.8).integrate(20_000, 0.01, seed=5)["x"], 0.01)
        for name in ("theta", "sigma"):
            assert fit.stderr[name] < 0.5 * fit.params[name], f"{name} stderr looks like the identity"

    def test_more_data_shrinks_the_error_bars(self):
        truth = OrnsteinUhlenbeck(theta=2.0, mu=0.5, sigma=1.0)
        long = truth.integrate(40_000, 0.01, seed=8)["x"]
        short_fit = fit_ornstein_uhlenbeck(long[:2_000], 0.01)
        long_fit = fit_ornstein_uhlenbeck(long, 0.01)
        assert long_fit.stderr["theta"] < short_fit.stderr["theta"]

    def test_too_little_data_refuses_to_fit(self):
        with pytest.raises(ValueError):
            fit_ornstein_uhlenbeck([1.0, 2.0, 3.0], dt=0.01)

    def test_impact_fit_recovers_lambda_and_flags_noise(self):
        rng = np.random.default_rng(4)
        q = rng.normal(0, 1, 3_000)
        real = fit_impact(q, 2.5 * q + rng.normal(0, 1, 3_000))
        assert real.params["lam"] == pytest.approx(2.5, rel=0.1)
        assert abs(real.params["lam"]) > 3 * real.stderr["lam"]        # clearly non-zero

        none = fit_impact(q, rng.normal(0, 1, 3_000))                  # no relationship at all
        assert abs(none.params["lam"]) < 3 * none.stderr["lam"], "found impact in pure noise"


# ── 3. does each model do the physical thing it claims? ───────────────────────

class TestMicrostructure:
    def test_the_mid_chases_the_efficient_price(self):
        path = OrderBookMicrostructure().integrate(4_000, 0.001, seed=3)
        assert np.corrcoef(path["E"], path["M"])[0, 1] > 0.9

    def test_the_mid_lags_rather_than_leads(self):
        """Roll's insight: observed mid is a noisy, delayed view of the efficient price."""
        path = OrderBookMicrostructure().integrate(20_000, 0.001, seed=3)
        e, m = path["E"], path["M"]
        lead = np.corrcoef(e[:-5], m[5:])[0, 1]       # E today vs M later
        lag = np.corrcoef(e[5:], m[:-5])[0, 1]        # M today vs E later
        assert lead > lag, "the efficient price must lead the observed mid"

    def test_spreads_widen_when_flow_is_imbalanced(self):
        path = OrderBookMicrostructure().integrate(20_000, 0.001, seed=6)
        assert np.corrcoef(np.abs(path["Q"]), path["S"])[0, 1] > 0.3

    def test_our_participation_moves_the_price_against_us(self):
        base = OrderBookMicrostructure()
        dw = base.draw_increments(3_000, 0.001, seed=9)
        passive = base.integrate(3_000, 0.001, increments=dw)
        from dataclasses import replace
        aggressive = replace(base, participation=0.30).integrate(3_000, 0.001, increments=dw)
        assert aggressive.final("M") > passive.final("M")

    def test_execution_cost_is_concave_in_size(self):
        """The square-root law: cost rises with size, cost PER UNIT falls."""
        from dataclasses import replace
        base = OrderBookMicrostructure()
        dw = base.draw_increments(2_000, 0.001, seed=3)
        sizes = [0.02, 0.05, 0.10, 0.25, 0.50]
        costs = []
        for u in sizes:
            proc = replace(base, participation=u)
            costs.append(proc.execution_cost_bps(proc.integrate(2_000, 0.001, increments=dw)))
        assert costs == sorted(costs), "total cost must rise with size"
        per_unit = [c / u for c, u in zip(costs, sizes)]
        assert per_unit == sorted(per_unit, reverse=True), "cost per unit must fall with size"

    def test_doing_nothing_costs_nothing(self):
        book = OrderBookMicrostructure(participation=0.0)
        assert book.execution_cost_bps(book.integrate(500, 0.001, seed=1)) == 0.0


class TestSupplyChain:
    def test_goods_ordered_today_arrive_after_the_delay_and_not_before(self):
        """
        The defining property. If arrivals respond to an order before tau has
        elapsed, the model has a causality violation and is worthless.

        The intervention is `order_gain` - how hard we order - because it changes
        the order rate from day zero while leaving the initial state untouched.
        (Intervening on target_inventory would also move the starting inventory,
        so the two runs would differ immediately for a reason that has nothing to
        do with delay.) Orders must diverge at once; inventory must not move until
        tau has passed.
        """
        from dataclasses import replace
        # Noise is required, not incidental: the model starts exactly at steady
        # state, where the base-stock gap is zero and order_gain multiplies
        # nothing. Demand has to move before how-hard-we-order can matter.
        chain = SupplyChainDelay(delay_days=45.0)
        eager = replace(chain, order_gain=0.9)
        dt, lag = 1.0, 45
        dw = chain.draw_increments(120, dt, seed=1)          # shared noise
        a = chain.integrate(120, dt, increments=dw)
        b = eager.integrate(120, dt, increments=dw)

        assert not np.allclose(a["O"][2:6], b["O"][2:6]), "ordering never changed at all"
        assert np.allclose(a["I"][:lag + 1], b["I"][:lag + 1], atol=1e-9), (
            "inventory responded before the delay elapsed - the model violates causality"
        )
        assert not np.allclose(a["I"][lag + 5:], b["I"][lag + 5:]), "the order never arrived"

    def test_pipeline_converges_to_delay_times_demand(self):
        """Little's law: in transit = delay x throughput. If it does not, the policy is wrong."""
        chain = SupplyChainDelay(sigma_d=0.0, sigma_i=0.0)
        path = chain.integrate(600, 1.0, seed=2)
        expected = chain.delay_days * chain.demand_base
        assert float(np.mean(path["O"][300:])) == pytest.approx(expected, rel=0.1)

    def test_inventory_does_not_drain_at_steady_state(self):
        """
        Regression: the base-stock policy ordered nothing at target, so inventory
        drained at the demand rate and went negative while the position looked fine.
        """
        chain = SupplyChainDelay(sigma_d=0.0, sigma_i=0.0)
        inventory = chain.integrate(1_000, 1.0, seed=1)["I"]
        assert inventory.min() > 0, "inventory went negative under a stable policy"
        assert float(inventory[-1]) == pytest.approx(chain.target_inventory, rel=0.4)

    def test_a_longer_delay_hurts_more_than_a_demand_shock(self):
        """Delay, not demand, is what breaks a supply chain. The model must say so."""
        from dataclasses import replace
        base = SupplyChainDelay()
        dw = base.draw_increments(400, 1.0, seed=5)
        worst = lambda p: float(p["I"].min())
        normal = worst(base.integrate(400, 1.0, increments=dw))
        demand_shock = worst(replace(base, demand_base=130.0).integrate(400, 1.0, increments=dw))
        delay_shock = worst(replace(base, delay_days=75.0).integrate(400, 1.0, increments=dw))
        assert delay_shock < demand_shock < normal

    def test_shortage_pressure_is_one_sided(self):
        chain = SupplyChainDelay()
        path = chain.integrate(300, 1.0, seed=3)
        pressure = chain.shortage_pressure_bps(path)
        assert np.all(pressure >= 0), "a glut must not produce negative pressure"
        assert pressure[np.argmin(path["I"])] == pressure.max()


# ── 4. counterfactual branching ───────────────────────────────────────────────

class TestBranching:
    def test_branches_share_noise_so_differences_are_causal(self):
        mv = SDEMultiverse(OrderBookMicrostructure(), "M").explore(
            {"a": {"participation": 0.0}, "b": {"participation": 0.2}},
            n_steps=500, dt=0.001, seed=3)
        a, b = mv.branches
        assert np.allclose(a.path.increments, b.path.increments), "branches must share noise"
        assert not np.allclose(a.path.states, b.path.states), "and still differ by the action"

    def test_a_null_intervention_reproduces_the_baseline_exactly(self):
        mv = SDEMultiverse(OrderBookMicrostructure(), "M").explore(
            {"nothing": {}}, n_steps=400, dt=0.001, seed=1)
        assert np.allclose(mv.branches[0].path.states, mv.baseline.states)

    def test_unknown_outcome_is_rejected(self):
        with pytest.raises(ValueError):
            SDEMultiverse(OrderBookMicrostructure(), "not_a_state")

    def test_report_labels_simulated_output(self):
        mv = SDEMultiverse(OrderBookMicrostructure(), "M").explore(
            {"x": {"participation": 0.1}}, n_steps=200, dt=0.001, seed=1)
        assert "SIMULATED" in mv.report()

    def test_the_discrete_scm_still_works(self):
        """Phase 2 added the SDE layer; it must not have broken the SCM under it."""
        g = (CausalGraph()
             .add("a", (), lambda v, e: 1.0 + e)
             .add("b", ("a",), lambda v, e: 2.0 * v["a"] + 0.1 * e))
        base = g.sample(n=500, seed=1)
        assert np.allclose(g.counterfactual(base, a=0.0)["a"], 0.0)


# ── 5. real-time prediction error ─────────────────────────────────────────────

def regime_shift_series(dt: float = 0.05, n: int = 600) -> np.ndarray:
    """A world that moves: OU reverting to 0, then to +3, with nobody announcing it."""
    first = OrnsteinUhlenbeck(theta=2.0, mu=0.0, sigma=0.5).integrate(n, dt, seed=4)["x"]
    second = OrnsteinUhlenbeck(theta=2.0, mu=3.0, sigma=0.5,
                               x0=float(first[-1])).integrate(n, dt, seed=5)["x"]
    return np.concatenate([first, second[1:]])


class TestRealTime:
    def test_an_sde_satisfies_the_predictive_protocol(self):
        assert isinstance(OrnsteinUhlenbeck(), PredictiveProcess)
        assert isinstance(OrderBookMicrostructure(), PredictiveProcess)

    def test_the_filter_beats_the_frozen_model_after_a_regime_shift(self):
        """
        The Phase 2 claim, stated as a falsifiable test: minimizing prediction
        error in real time beats a static fit when the world moves.
        """
        results = compare_online_vs_static(
            OrnsteinUhlenbeck(theta=2.0, mu=0.0, sigma=0.5), regime_shift_series(), dt=0.05)
        static, online = results["static"], results["online"]
        assert online.rmse < static.rmse
        assert online.cumulative_free_energy < static.cumulative_free_energy

    def test_the_filter_does_not_hurt_when_the_model_is_already_right(self):
        """Adaptation must not cost much when there is nothing to adapt to."""
        stable = OrnsteinUhlenbeck(theta=2.0, mu=0.0, sigma=0.5)
        series = stable.integrate(1_200, 0.05, seed=7)["x"]
        results = compare_online_vs_static(stable, series, dt=0.05)
        assert results["online"].rmse < results["static"].rmse * 1.15

    def test_a_persistent_error_is_reported_as_drift(self):
        biased = PredictionErrorLoop(OrnsteinUhlenbeck(theta=2.0, mu=0.0, sigma=0.5),
                                     dt=0.05, adapt=False)
        biased.run(regime_shift_series())
        assert biased.result.drifting(), "a one-directional error run must be flagged"

    def test_noise_alone_is_not_reported_as_drift(self):
        stable = OrnsteinUhlenbeck(theta=2.0, mu=0.0, sigma=0.5)
        loop = PredictionErrorLoop(stable, dt=0.05, adapt=False)
        loop.run(stable.integrate(1_200, 0.05, seed=12)["x"])
        assert not loop.result.drifting(), "false alarm on a model that is fine"

    def test_the_filter_learns_the_correct_residual_bias(self):
        """
        Offset every observation by +5 and the filter must learn the bias that a
        one-step-ahead OU prediction actually needs - which is NOT +5.

        The prediction starts from the observed state, and that state already
        carries the offset. What survives one step is the mean-reversion pull
        back toward the model's mu: 5 * (1 - exp(-theta*dt)) = 0.2439 here. A
        filter that learned +5 would be double-counting the shift.
        """
        theta, dt = 1.0, 0.05
        stable = OrnsteinUhlenbeck(theta=theta, mu=0.0, sigma=0.2)
        series = stable.integrate(2_000, dt, seed=2)["x"] + 5.0
        loop = PredictionErrorLoop(stable, dt=dt, adapt=True)
        loop.run(series)
        expected = 5.0 * (1.0 - np.exp(-theta * dt))
        assert loop.result.final_bias == pytest.approx(expected, rel=0.15)

    def test_calibration_is_near_one_when_the_model_is_correct(self):
        """If stated uncertainty is honest, standardised errors have unit variance."""
        stable = OrnsteinUhlenbeck(theta=2.0, mu=0.0, sigma=0.5)
        loop = PredictionErrorLoop(stable, dt=0.02, adapt=False)
        loop.run(stable.integrate(4_000, 0.02, seed=15)["x"])
        assert 0.85 < loop.result.calibration < 1.15

    def test_the_loop_is_incremental_not_batch(self):
        """
        Real time means one observation at a time. step() and run() must agree,
        or the 'real-time' claim is a batch process wearing a costume.
        """
        stable = OrnsteinUhlenbeck(theta=2.0, mu=0.0, sigma=0.5)
        series = stable.integrate(200, 0.05, seed=3)["x"]
        batch = PredictionErrorLoop(stable, dt=0.05).run(series)
        incremental = PredictionErrorLoop(stable, dt=0.05)
        for i in range(len(series) - 1):
            incremental.step(series[i], series[i + 1])
        assert incremental.result.rmse == pytest.approx(batch.rmse, rel=1e-12)

    def test_a_single_observation_cannot_be_filtered(self):
        with pytest.raises(ValueError):
            PredictionErrorLoop(OrnsteinUhlenbeck(), dt=0.05).run([1.0])

    def test_free_energy_accounting_is_shared_with_the_offline_loop(self):
        loop = PredictionErrorLoop(OrnsteinUhlenbeck(theta=2.0, mu=0.0, sigma=0.5), dt=0.05)
        loop.run(regime_shift_series())
        assert isinstance(loop.inference, ActiveInferenceLoop)
        assert len(loop.inference.history) == loop.result.n
