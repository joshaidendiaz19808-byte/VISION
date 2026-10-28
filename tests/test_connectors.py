"""Connectors against canned provider responses (no network, no real keys).
The important property: a key never shows up in an error message."""
import json

import pytest
import requests

from checks import connectors


class FakeResponse:
    def __init__(self, body, status=200, ctype="application/json"):
        self.body = body.encode() if isinstance(body, str) else body
        self.status_code = status
        self.headers = {"content-type": ctype}

    def iter_content(self, n):
        yield self.body


@pytest.fixture
def serve(monkeypatch):
    """serve([...responses]) - each request pops the next response and is recorded."""
    calls = []

    def install(responses):
        def fake(method, url, headers=None, **kw):
            calls.append({"method": method, "url": url, "headers": headers, **kw})
            return responses.pop(0)
        monkeypatch.setattr(requests, "request", fake)
        monkeypatch.setattr(connectors.time, "sleep", lambda s: None)
        return calls
    return install


SECRET = "sk-SUPER-SECRET-123"


def test_errors_never_contain_the_key(serve):
    for status in (401, 403, 404, 500):
        serve([FakeResponse("", status)])
        with pytest.raises(Exception) as e:
            connectors.ibkr_flex_trades(SECRET, "999")
        assert SECRET not in str(e.value)


def test_network_error_never_contains_the_key(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError(f"failed for https://x.com/?t={SECRET}")
    monkeypatch.setattr(requests, "request", boom)
    with pytest.raises(ConnectionError) as e:
        connectors.ibkr_flex_trades(SECRET, "1")
    assert SECRET not in str(e.value)


IBKR_SEND = """<FlexStatementResponse><Status>Success</Status><ReferenceCode>REF42</ReferenceCode>
<Url>https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/GetStatement</Url></FlexStatementResponse>"""
IBKR_WAIT = """<FlexStatementResponse><Status>Warn</Status><ErrorCode>1019</ErrorCode>
<ErrorMessage>Statement generation in progress.</ErrorMessage></FlexStatementResponse>"""
IBKR_STMT = """<FlexQueryResponse queryName="vision" type="AF"><FlexStatements count="1"><FlexStatement>
<Trades>
 <Trade symbol="ES" tradeDate="20260105" cost="10000" fifoPnlRealized="0"/>
 <Trade symbol="ES" tradeDate="20260106" cost="10000" fifoPnlRealized="150"/>
 <Trade symbol="NQ" tradeDate="20260102" cost="-20000" fifoPnlRealized="-100"/>
</Trades></FlexStatement></FlexStatements></FlexQueryResponse>"""


def test_ibkr_polls_then_parses_closed_trades(serve):
    calls = serve([FakeResponse(IBKR_SEND, ctype="text/xml"), FakeResponse(IBKR_WAIT, ctype="text/xml"),
                   FakeResponse(IBKR_STMT, ctype="text/xml")])
    df = connectors.ibkr_flex_trades(SECRET, "123")
    assert list(df["symbol"]) == ["NQ", "ES"]  # opening trade dropped, sorted by date
    assert df["return_pct"].tolist() == pytest.approx([-0.5, 1.5])
    assert calls[1]["params"]["q"] == "REF42" and len(calls) == 3
    assert all(SECRET not in c["url"] for c in calls)  # token only ever in params


def test_ibkr_reports_refusal(serve):
    serve([FakeResponse("<FlexStatementResponse><Status>Fail</Status><ErrorCode>1012</ErrorCode>"
                        "<ErrorMessage>Token has expired.</ErrorMessage></FlexStatementResponse>")])
    with pytest.raises(ConnectionError, match="Token has expired"):
        connectors.ibkr_flex_trades(SECRET, "1")


def _run(i, acc):
    return {"info": {"run_id": f"r{i}", "status": "FINISHED", "start_time": 1767225600000 + i},
            "data": {"metrics": [{"key": "accuracy", "value": acc}], "params": [{"key": "depth", "value": str(i)}]}}


def test_mlflow_pages_through_every_run(serve):
    calls = serve([
        FakeResponse(json.dumps({"runs": [_run(i, 0.6 + i / 100) for i in range(3)], "next_page_token": "p2"})),
        FakeResponse(json.dumps({"runs": [_run(i, 0.6) for i in range(3, 5)]})),
    ])
    df = connectors.mlflow_runs("https://dbc.cloud.databricks.com/", ["7"], token=SECRET)
    assert len(df) == 5 and {"metric.accuracy", "param.depth"} <= set(df.columns)
    assert calls[0]["url"] == "https://dbc.cloud.databricks.com/api/2.0/mlflow/runs/search"
    assert calls[1]["json"]["page_token"] == "p2"
    assert calls[0]["headers"]["Authorization"] == f"Bearer {SECRET}"
