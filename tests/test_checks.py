"""Every check is tested on planted truth: a known-fake input must be caught,
a known-clean input must pass. A detector that only ever says FAKE is useless."""
import json

import numpy as np
import pandas as pd
import pytest

from checks import ab_test, code_audit, integrity, ml_model, trading, detect, FAIL, PASS, WARN


def status(rep, name):
    return next(c.status for c in rep["checks"] if c.name.startswith(name))


# ── trading ─────────────────────────────────────────────────────────────────

def test_trading_real_edge_passes_and_noise_fails():
    rng = np.random.default_rng(1)
    real = trading.run(pd.Series(rng.normal(0.3, 0.8, 400)))
    noise = trading.run(pd.Series(rng.normal(0.0, 0.8, 400)))
    assert real["verdict"] == "LOOKS REAL"
    assert noise["verdict"] == "LOOKS FAKE"


def test_deflated_sharpe_punishes_many_tries():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.08, 1.0, 250))  # a modest edge
    assert status(trading.run(r, n_trials=1), "Tried many") == "skip"
    assert status(trading.run(r, n_trials=1000), "Tried many") == FAIL


def test_smoothed_returns_flagged():
    rng = np.random.default_rng(3)
    e = rng.normal(0.1, 1, 500)
    smooth = pd.Series(np.convolve(e, np.ones(5) / 5, mode="same"))
    assert status(trading.run(smooth), "Returns too smooth") == WARN
    assert status(trading.run(pd.Series(e)), "Returns too smooth") == PASS


# ── A/B tests ────────────────────────────────────────────────────────────────

def _ab(n_a, n_b, p_a, p_b, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "variant": ["control"] * n_a + ["B"] * n_b,
        "converted": np.r_[rng.random(n_a) < p_a, rng.random(n_b) < p_b].astype(int),
        "ts": np.r_[np.arange(n_a), np.arange(n_b)],
    })


def test_ab_real_lift_passes():
    rep = ab_test.run(_ab(20000, 20000, 0.10, 0.12), "variant", "converted", time_col="ts")
    assert status(rep, "Difference is real") == PASS
    assert status(rep, "Groups split") == PASS


def test_ab_no_lift_fails():
    rep = ab_test.run(_ab(20000, 20000, 0.10, 0.10), "variant", "converted")
    assert status(rep, "Difference is real") == FAIL


def test_ab_sample_ratio_mismatch_caught():
    rep = ab_test.run(_ab(20000, 18500, 0.10, 0.12), "variant", "converted")
    assert status(rep, "Groups split") == FAIL


def test_ab_multiple_metrics_correction():
    # exactly 10.0% vs 11.4% conversion: significant alone, not after testing 40 metrics
    df = pd.DataFrame({"variant": ["control"] * 5000 + ["B"] * 5000,
                       "converted": [1] * 500 + [0] * 4500 + [1] * 570 + [0] * 4430})
    p = ab_test.run(df, "variant", "converted")["checks"][2].detail["p"]
    assert 0.001 < p < 0.05
    assert status(ab_test.run(df, "variant", "converted", n_metrics=40), "Difference is real") == FAIL


def test_ab_whale_driven_revenue_caught():
    rng = np.random.default_rng(5)
    a = rng.exponential(10, 3000)
    b = rng.exponential(10, 3000)
    b[:6] = 20000  # six whales
    df = pd.DataFrame({"group": ["A"] * 3000 + ["B"] * 3000, "revenue": np.r_[a, b]})
    rep = ab_test.run(df, "group", "revenue")
    assert status(rep, "Difference is real") == PASS
    assert status(rep, "Not driven by a few") == FAIL


# ── ML models ────────────────────────────────────────────────────────────────

def _ml(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + rng.normal(size=n) > 0).astype(int)
    df = pd.DataFrame(x, columns=["f1", "f2", "f3"]).assign(y=y, date=pd.date_range("2020", periods=n, freq="D"))
    return df


def test_ml_clean_split_passes():
    df = _ml()
    rep = ml_model.run(train=df.iloc[:1500], test=df.iloc[1500:], target="y", time_col="date")
    assert rep["verdict"] == "LOOKS REAL", [c for c in rep["checks"] if c.status != PASS]


def test_ml_leaky_column_overlap_and_time_caught():
    df = _ml()
    df["refund_issued"] = df["y"]  # known only after the outcome
    shuffled = df.sample(frac=1, random_state=0)
    train, test = shuffled.iloc[:1500], pd.concat([shuffled.iloc[1500:], shuffled.iloc[:100]])
    rep = ml_model.run(train=train, test=test, target="y", time_col="date")
    assert status(rep, "No column gives away") == FAIL
    assert "refund_issued" in rep["checks"][-1].detail["suspects"]
    assert status(rep, "Test rows not in training") == FAIL
    assert status(rep, "Tested on the future") == FAIL


def test_ml_predictions_vs_baseline():
    rng = np.random.default_rng(6)
    y = (rng.random(1000) < 0.9).astype(int)  # 90% one class
    useless = pd.DataFrame({"y_true": y, "y_pred": 1})
    good = pd.DataFrame({"y_true": y, "y_pred": np.where(rng.random(1000) < 0.98, y, 1 - y)})
    assert status(ml_model.run(pred=useless, y_true="y_true", y_pred="y_pred"), "Better than a dumb") == FAIL
    assert status(ml_model.run(pred=good, y_true="y_true", y_pred="y_pred"), "Better than a dumb") == PASS


# ── data integrity ───────────────────────────────────────────────────────────

def test_benford_natural_vs_invented():
    rng = np.random.default_rng(7)
    natural = pd.DataFrame({"amount": np.round(np.exp(rng.uniform(0, 9, 3000)), 2)})
    invented = pd.DataFrame({"amount": np.round(rng.uniform(1, 100000, 3000), 2)})  # flat, not natural
    assert status(integrity.run(natural), "'amount' first digits") == PASS
    assert status(integrity.run(invented), "'amount' first digits") == WARN


def test_last_digit_preference():
    rng = np.random.default_rng(8)
    honest = pd.Series(rng.integers(100, 9999, 2000)).astype(str)
    fudged = pd.Series(rng.integers(10, 999, 2000) * 10 + rng.choice([0, 5], 2000)).astype(str)
    assert integrity.last_digit("x", honest).status == PASS
    assert integrity.last_digit("x", fudged).status == WARN


def test_p_hacking_bump():
    rng = np.random.default_rng(9)
    hacked = pd.Series(np.r_[rng.uniform(0.045, 0.05, 40), rng.uniform(0.04, 0.045, 8)])
    honest = pd.Series(np.r_[rng.uniform(0.045, 0.05, 10), rng.uniform(0.04, 0.045, 14)])
    assert integrity.p_hacking(hacked).status == FAIL
    assert integrity.p_hacking(honest).status == PASS


def test_grim_impossible_means():
    df = pd.DataFrame({"mean": ["3.48", "3.50", "2.25"], "n": [20, 20, 12]})
    # 3.48 * 20 = 69.6 -> not a whole number; 3.50 and 2.25 are fine
    assert integrity.grim(df, "mean", "n").status == FAIL
    assert integrity.grim(df.iloc[1:], "mean", "n").status == PASS


# ── code audit ───────────────────────────────────────────────────────────────

LEAKY = '''
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv("prices.csv", parse_dates=["date"])
df["target"] = df["close"].shift(-1) > df["close"]
df["momentum"] = df["close"].shift(-5) / df["close"]
df["smooth"] = df["close"].rolling(20, center=True).mean()
df = df.bfill()
df["z"] = (df["close"] - df["close"].mean()) / df["close"].std()
scaler = StandardScaler()
X = scaler.fit_transform(df[["momentum", "smooth", "z"]])
X_train, X_test, y_train, y_test = train_test_split(X, df["target"])
best = 0
for depth in range(1, 20):
    m = Model(depth).fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    if acc > best:
        best = acc
for seed in range(100):
    m = Model(random_state=seed).fit(X_train, y_train)
    s = m.score(X_test, y_test)
    best_s = max(best_s, s)
print(accuracy_score(y_train, m.predict(X_train)))
for i in range(len(df) - 1):
    if df.close[i + 1] > df.close[i]:
        position = 1
pnl = (df.close.diff() * position).cumsum()
'''

CLEAN = '''
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv("prices.csv", parse_dates=["date"])
df["target"] = df["close"].shift(-1) > df["close"]
df["momentum"] = df["close"] / df["close"].shift(5)
df["z"] = (df["close"] - df["close"].rolling(50).mean()) / df["close"].rolling(50).std()
df = df.ffill().dropna()
X_train, X_test, y_train, y_test = train_test_split(df[["momentum", "z"]], df["target"], shuffle=False)
model = make_pipeline(StandardScaler(), Model(random_state=0)).fit(X_train, y_train)
print(model.score(X_test, y_test))
fee_bps = 2
for i in range(1, len(df)):
    if df.signal[i - 1] > 0:
        pnl = df.close[i + 1] - df.close[i] - fee_bps
'''


def test_code_audit_catches_every_planted_bug(tmp_path):
    (tmp_path / "leaky.py").write_text(LEAKY)
    rep = code_audit.run(tmp_path)
    failed = {c.name for c in rep["checks"] if c.status in (FAIL, WARN)}
    for rule in ("future_shift", "centered_window", "backfill", "full_sample_norm", "fit_before_split",
                 "shuffled_time_split", "test_set_selection", "seed_fishing", "train_score",
                 "next_bar_signal", "no_costs"):
        assert code_audit.RULES[rule][0] in failed, rule
    assert rep["verdict"] == "LOOKS FAKE"


def test_code_audit_clean_pipeline_passes(tmp_path):
    (tmp_path / "clean.py").write_text(CLEAN)
    rep = code_audit.run(tmp_path)
    flagged = [(c.name, c.reason) for c in rep["checks"] if c.status != PASS]
    assert not flagged, flagged


def test_code_audit_unseeded_randomness(tmp_path):
    (tmp_path / "a.py").write_text("import numpy as np\nw = np.random.normal(size=10)\n")
    assert status(code_audit.run(tmp_path), code_audit.RULES["unseeded"][0]) == WARN
    (tmp_path / "a.py").write_text("import numpy as np\nnp.random.seed(0)\nw = np.random.normal(size=10)\n")
    assert status(code_audit.run(tmp_path), code_audit.RULES["unseeded"][0]) == PASS


def test_code_audit_reads_notebooks_and_order(tmp_path):
    nb = {"cells": [
        {"cell_type": "code", "execution_count": 3, "source": ["%matplotlib inline\n", "import pandas as pd\n"]},
        {"cell_type": "code", "execution_count": 1, "source": ["df['x'] = df['close'].shift(-2)\n"]},
    ]}
    (tmp_path / "a.ipynb").write_text(json.dumps(nb))
    rep = code_audit.run(tmp_path)
    assert status(rep, code_audit.RULES["notebook_order"][0]) == WARN
    assert status(rep, code_audit.RULES["future_shift"][0]) == FAIL


def test_code_audit_never_executes(tmp_path):
    marker = tmp_path / "pwned"
    (tmp_path / "evil.py").write_text(f"open({str(marker)!r}, 'w').write('x')\n")
    code_audit.run(tmp_path)
    assert not marker.exists()


# ── detection ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cols,kind", [
    (["date", "return_pct"], "trading"),
    (["user", "variant", "converted"], "ab"),
    (["id", "y_true", "y_pred"], "predictions"),
    (["patient", "dose", "response"], "data"),
])
def test_detect(cols, kind):
    df = pd.DataFrame({c: ([0.1, -0.2] * 5 if c != "variant" else ["A", "B"] * 5) for c in cols})
    assert detect.guess(df)[0] == kind
