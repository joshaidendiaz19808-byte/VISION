"""Trading strategy results: the original four honesty tests plus a multiple-testing check.

Input: per-trade (or per-day) % returns, in order. 0.5 == a 0.5% gain.
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.vision import (
    test_sample_size, test_out_of_sample, test_remove_best, test_cost_stress, test_monte_carlo,
)
from .common import Check, report, PASS, FAIL, SKIP, WARN


def _wrap(name, out):
    return Check(name, PASS if out["passed"] else FAIL, out["reason"] or "", out["fix"] or "",
                 {k: v for k, v in out.items() if k not in ("passed", "reason", "fix")})


def deflated_sharpe(returns: pd.Series, n_trials: int) -> Check:
    """Bailey & Lopez de Prado: is the best of N tried versions better than the luckiest noise would be?"""
    name = "Tried many versions"
    if n_trials <= 1:
        return Check(name, SKIP, "Tell VISION how many versions / settings you tried before this one. "
                     "Trying 50 versions and keeping the best one is the #1 way backtests lie.")
    r = returns.to_numpy(dtype=float)
    t = len(r)
    if t < 30 or r.std(ddof=1) == 0:
        return Check(name, SKIP, "Too few trades to adjust for multiple tries.")
    sr = r.mean() / r.std(ddof=1)
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    euler = 0.5772156649
    sr_sd = np.sqrt(1.0 / (t - 1))  # spread of Sharpe estimates across pure-noise trials
    sr0 = sr_sd * ((1 - euler) * stats.norm.ppf(1 - 1 / n_trials)
                   + euler * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    dsr = float(stats.norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom))
    detail = {"deflated_sharpe_prob": dsr, "n_trials": n_trials}
    if dsr >= 0.95:
        return Check(name, PASS, detail=detail)
    return Check(
        name, FAIL,
        f"You tried {n_trials} versions. The best of {n_trials} random strategies would look about this "
        f"good by luck alone - only a {dsr*100:.0f}% chance this one is genuinely better than noise.",
        "Lock your settings, then test once on data you have never looked at. Fewer tries = more believable.",
        detail,
    )


def serial_dependence(returns: pd.Series) -> Check:
    """Smoothed/marked-to-model returns show strong positive autocorrelation and understate risk."""
    name = "Returns too smooth"
    r = returns.to_numpy(dtype=float)
    if len(r) < 50:
        return Check(name, SKIP, "Too few returns to check smoothing.")
    rho = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    se = 1 / np.sqrt(len(r))
    detail = {"lag1_autocorr": rho}
    if rho > max(0.3, 3 * se):
        return Check(
            name, WARN,
            f"Each return looks a lot like the one before it (autocorrelation {rho:+.2f}). Real trade results "
            "rarely do that - it usually means returns are smoothed, marked to model, or overlap.",
            "Use actual fill prices and non-overlapping holding periods. Smoothed returns hide the real risk.",
            detail,
        )
    return Check(name, PASS, detail=detail)


def run(returns: pd.Series, cost_bps: float = 5.0, n_trials: int = 1) -> dict:
    returns = pd.to_numeric(returns, errors="coerce").dropna().reset_index(drop=True)
    checks = [
        _wrap("Enough trades", test_sample_size(returns)),
        _wrap("Still works on recent data", test_out_of_sample(returns)),
        _wrap("Not just a few lucky wins", test_remove_best(returns)),
        _wrap("Survives trading costs", test_cost_stress(returns, cost_bps=cost_bps)),
        _wrap("Not luck (Monte Carlo)", test_monte_carlo(returns)),
        deflated_sharpe(returns, n_trials),
        serial_dependence(returns),
    ]
    return report("Trading strategy", checks, n=len(returns), unit="trades")
