"""Every API endpoint the web and phone apps call, end to end through HTTP."""
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import server
from checks import connectors

client = TestClient(server.app)


def csv(df):
    return df.to_csv(index=False).encode()


def post_table(df, kind="auto", options=None, name="r.csv", data=None):
    return client.post("/api/check", data={"kind": kind, "options": json.dumps(options or {})},
                       files={"file": (name, data if data is not None else csv(df))})


def test_health():
    assert client.get("/api/health").json() == {"ok": True}


def test_trading_auto_detected_and_verdict():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"date": pd.date_range("2020", periods=300).astype(str),
                       "return_pct": rng.normal(0.3, 0.8, 300)})
    r = post_table(df).json()
    assert r["kind"] == "trading" and r["chosen"]["returns"] == "return_pct"
    assert r["report"]["verdict"] == "LOOKS REAL"
    assert {"name", "status", "reason", "fix", "detail"} <= set(r["report"]["checks"][0])
    json.dumps(r)  # fully serializable


def test_trading_options_respected():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(0.08, 1, 250), "b": rng.normal(0, 1, 250)})
    r = post_table(df, "trading", {"returns": "a", "n_trials": 1000}).json()
    assert r["chosen"]["returns"] == "a"
    tried = next(c for c in r["report"]["checks"] if c["name"] == "Tried many versions")
    assert tried["status"] == "fail"


def test_dollar_column_rejected_with_reason():
    df = pd.DataFrame({"pnl": np.linspace(-500, 900, 120)})
    r = post_table(df, "trading")
    assert r.status_code == 400 and "dollar" in r.json()["detail"]


def test_unknown_column_rejected():
    r = post_table(pd.DataFrame({"x": [1.0, 2.0]}), "trading", {"returns": "nope"})
    assert r.status_code == 400 and "nope" in r.json()["detail"]


def test_ab_and_groups_listed():
    df = pd.DataFrame({"variant": ["control"] * 5000 + ["B"] * 5000,
                       "converted": [1] * 500 + [0] * 4500 + [1] * 700 + [0] * 4300})
    r = post_table(df).json()
    assert r["kind"] == "ab" and r["chosen"]["groups"] == ["B", "control"]
    assert r["report"]["verdict"] in ("LOOKS REAL", "QUESTIONABLE")


def test_excel_upload():
    buf = io.BytesIO()
    pd.DataFrame({"y_true": [0, 1] * 300, "y_pred": [0, 1] * 300}).to_excel(buf, index=False)
    r = post_table(None, name="p.xlsx", data=buf.getvalue()).json()
    assert r["kind"] == "predictions"


def test_garbage_and_empty_files():
    assert post_table(None, name="x.xlsx", data=b"not excel").status_code == 400
    assert post_table(None, data=b"a,b\n").status_code == 400
    assert client.post("/api/check", data={"kind": "auto"}).status_code == 400


def test_ml_leakage_endpoint():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"f": rng.normal(size=600)})
    df["y"] = (df.f > 0).astype(int)
    df["leak"] = df["y"]
    r = client.post("/api/check/ml", data={"options": json.dumps({"target": "y"})},
                    files={"train": ("tr.csv", csv(df[:400])), "test": ("te.csv", csv(df[400:]))}).json()
    assert r["report"]["verdict"] == "LOOKS FAKE"
    assert r["columns"]["train"] == ["f", "y", "leak"]


def test_ml_needs_inputs():
    assert client.post("/api/check/ml", data={}).status_code == 400


def test_code_upload_audit_and_ai_needs_key(tmp_path):
    src = b'import pandas as pd\ndf = pd.read_csv("p.csv")\ndf["x"] = df["close"].shift(-3)\n'
    r = client.post("/api/code", files=[("files", ("s.py", src))]).json()
    assert r["report"]["verdict"] == "LOOKS FAKE" and r["code_id"]
    ai = client.post("/api/ai-review", json={"code_id": r["code_id"]})
    assert ai.status_code == 400 and "Anthropic" in ai.json()["detail"]
    assert client.post("/api/ai-review", json={"code_id": "nope", "anthropic_key": "k"}).status_code == 404


def test_zip_bomb_refused(monkeypatch):
    monkeypatch.setattr(server, "MAX_UNZIPPED", 1000)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("big.py", "x = 1\n" * 10000)
    r = client.post("/api/code", files=[("files", ("c.zip", buf.getvalue()))])
    assert r.status_code == 413


def test_code_rejects_other_files():
    assert client.post("/api/code", files=[("files", ("a.exe", b"MZ"))]).status_code == 400
    assert client.post("/api/code", files=[("files", ("a.zip", b"nope"))]).status_code == 400


def test_connect_url_and_private_network_block(monkeypatch):
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"return_pct": rng.normal(0.3, 0.8, 200)})
    monkeypatch.setattr(connectors, "url_table", lambda *a: df)
    r = client.post("/api/connect", json={"provider": "url", "credentials": {"url": "https://x.com/a.csv"}})
    assert r.json()["kind"] == "trading"
    monkeypatch.undo()
    monkeypatch.setattr(connectors.socket, "getaddrinfo", lambda *a: [(0, 0, 0, "", ("169.254.169.254", 0))])
    r = client.post("/api/connect", json={"provider": "url", "credentials": {"url": "http://metadata.internal/"}})
    assert r.status_code == 400 and "private" in r.json()["detail"]


def test_connect_bad_provider():
    assert client.post("/api/connect", json={"provider": "nope"}).status_code == 400


def test_combine_merges_reports():
    rng = np.random.default_rng(5)
    good = post_table(pd.DataFrame({"return_pct": rng.normal(0.3, 0.8, 300)})).json()["report"]
    bad = client.post("/api/code", files=[("files", ("s.py", b"df['x']=df['c'].rolling(5, center=True).mean()\n"))]).json()["report"]
    r = client.post("/api/combine", json={"reports": [good, bad]}).json()
    assert r["verdict"] == "LOOKS FAKE" and len(r["checks"]) == len(good["checks"]) + len(bad["checks"])


def test_cors_allows_phone_apps():
    for origin in ("capacitor://localhost", "https://localhost"):
        r = client.options("/api/health", headers={"Origin": origin, "Access-Control-Request-Method": "GET"})
        assert r.headers.get("access-control-allow-origin") == origin
    r = client.options("/api/health", headers={"Origin": "https://evil.com", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in r.headers


@pytest.fixture
def locked(monkeypatch):
    monkeypatch.setattr(server.app.state, "access_codes", ["alpha-123", "beta-456"])
    monkeypatch.setattr(server.app.state, "limits", {"check": (3, 3600), "ai": (1, 86400), "bad_code": (2, 3600)})


def test_access_code_required_when_set(locked):
    df = pd.DataFrame({"return_pct": [0.1, -0.2, 0.3] * 20})
    assert client.get("/api/health").status_code == 200  # health stays open for the host's checks
    r = post_table(df)
    assert r.status_code == 401 and r.json()["detail"] == "Enter your access code."
    r = client.post("/api/check", headers={"X-Vision-Code": "nope"}, files={"file": ("r.csv", csv(df))})
    assert r.status_code == 401 and "isn't valid" in r.json()["detail"]
    r = client.post("/api/check", headers={"X-Vision-Code": "beta-456"}, files={"file": ("r.csv", csv(df))})
    assert r.status_code == 200 and r.json()["kind"] == "trading"
    assert client.get("/").status_code == 200  # the web page itself loads without a code


def test_wrong_codes_get_locked_out(locked):
    for _ in range(2):
        assert client.post("/api/combine", headers={"X-Vision-Code": "guess"}, json={}).status_code == 401
    r = client.post("/api/combine", headers={"X-Vision-Code": "alpha-123x"}, json={})
    assert r.status_code == 429


def test_usage_limits_per_code(locked):
    ok = {"X-Vision-Code": "alpha-123"}
    for _ in range(3):
        assert client.post("/api/combine", headers=ok, json={"reports": []}).status_code == 200
    r = client.post("/api/combine", headers=ok, json={"reports": []})
    assert r.status_code == 429 and "Usage limit" in r.json()["detail"] and int(r.headers["Retry-After"]) > 0
    # another person's code has its own allowance
    assert client.post("/api/combine", headers={"X-Vision-Code": "beta-456"}, json={"reports": []}).status_code == 200
    # AI reviews have their own, tighter daily allowance
    assert client.post("/api/ai-review", headers=ok, json={"code_id": "x"}).status_code == 404
    r = client.post("/api/ai-review", headers=ok, json={"code_id": "x"})
    assert r.status_code == 429 and "h." in r.json()["detail"]


def test_limit_by_ip_when_open(monkeypatch):
    monkeypatch.setattr(server.app.state, "limits", {"check": (2, 3600), "ai": (1, 86400), "bad_code": (2, 3600)})
    assert server.app.state.access_codes == []
    codes = [client.post("/api/combine", json={"reports": []}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_cors_headers_on_refusals(locked):
    """The phone apps run on another origin: they must be able to read a 401 to show the code box."""
    r = client.post("/api/combine", headers={"Origin": "capacitor://localhost"}, json={})
    assert r.status_code == 401 and r.headers["access-control-allow-origin"] == "capacitor://localhost"
    pre = client.options("/api/check", headers={"Origin": "capacitor://localhost",
                                                "Access-Control-Request-Method": "POST",
                                                "Access-Control-Request-Headers": "x-vision-code"})
    assert pre.status_code == 200
