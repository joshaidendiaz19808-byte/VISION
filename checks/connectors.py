"""Pull results straight from where people keep them, using their own API access.

Keys are passed in per call and only ever live in memory: nothing here writes
a key to disk, a log or an error message.
"""
import io
import ipaddress
import json
import os
import socket
import time
import xml.etree.ElementTree as ET
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
import requests

TIMEOUT = 60
MAX_DOWNLOAD = 300 * 1024 * 1024


def check_public_url(url: str):
    """User-typed URLs must point at the public internet: a hosted server must not be talked into fetching
    its own private network (cloud metadata, internal services). VISION_ALLOW_PRIVATE_URLS=1 for local dev."""
    from urllib.parse import urlparse
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("Enter a full http(s) URL.")
    if os.environ.get("VISION_ALLOW_PRIVATE_URLS") == "1":
        return
    try:
        addrs = {a[4][0] for a in socket.getaddrinfo(u.hostname, u.port or 443)}
    except socket.gaierror:
        raise ConnectionError(f"Couldn't find {u.hostname}.") from None
    for a in addrs:
        ip = ipaddress.ip_address(a.split("%")[0])
        if not ip.is_global:
            raise PermissionError(f"{u.hostname} points at a private network address - VISION only fetches "
                                  "public URLs.")


def _request(method, url, headers=None, **kw):
    """Fetch with a size cap. Secrets go in headers/params, and errors only ever name the host + path,
    so a key can't leak into a message, a log or the page."""
    where = url.split("?")[0]
    try:
        r = requests.request(method, url, headers=headers or {}, timeout=TIMEOUT, stream=True, **kw)
    except requests.RequestException as e:
        raise ConnectionError(f"Couldn't reach {where.split('/')[2]} ({type(e).__name__}).") from None
    if r.status_code in (401, 403):
        raise PermissionError(f"Access denied by {where.split('/')[2]} - check the key has read permission.")
    if r.status_code == 404:
        raise FileNotFoundError(f"Not found: {where} (private? then add a key)")
    if 300 <= r.status_code < 400:
        raise ConnectionError(f"{where.split('/')[2]} tried to redirect - use the final URL.")
    if r.status_code >= 400:
        raise ConnectionError(f"{where.split('/')[2]} returned HTTP {r.status_code}.")
    buf = io.BytesIO()
    for chunk in r.iter_content(1 << 20):
        buf.write(chunk)
        if buf.tell() > MAX_DOWNLOAD:
            raise ValueError("Download is over 300 MB - point VISION at a smaller export.")
    return buf.getvalue(), r.headers.get("content-type", "")


def _get(url, headers=None, **kw):
    return _request("GET", url, headers, **kw)


def github_repo(repo: str, token: str | None = None, ref: str = "HEAD") -> Path:
    """Download a GitHub repo (owner/name or URL) into a temp folder for the code audit. Never executed."""
    repo = repo.strip().removeprefix("https://").removeprefix("http://").removeprefix("github.com/")
    repo = repo.removesuffix(".git").strip("/")
    owner, name = repo.split("/")[:2]
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data, _ = _get(f"https://api.github.com/repos/{owner}/{name}/tarball/{ref}", headers)
    dest = Path(tempfile.mkdtemp(prefix="vision_repo_"))
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")  # blocks path traversal, links and device files
    inner = [p for p in dest.iterdir() if p.is_dir()]
    return inner[0] if len(inner) == 1 else dest


def url_table(url: str, auth_header: str | None = None, auth_value: str | None = None,
              json_path: str | None = None) -> pd.DataFrame:
    """Any HTTP endpoint that returns CSV or JSON - a broker export, a lab API, an analytics API."""
    check_public_url(url)
    headers = {auth_header: auth_value} if auth_header and auth_value else {}
    data, ctype = _get(url, headers, allow_redirects=False)
    if "json" in ctype or url.endswith(".json") or data.lstrip()[:1] in (b"{", b"["):
        obj = json.loads(data)
        for key in filter(None, (json_path or "").split(".")):
            obj = obj[key]
        return pd.json_normalize(obj)
    return pd.read_csv(io.BytesIO(data))


def alpaca_returns(key_id: str, secret: str, paper: bool = True) -> pd.DataFrame:
    """Daily % returns of an Alpaca account (live or paper), from its portfolio history."""
    base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
    headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
    data, _ = _get(f"{base}/v2/account/portfolio/history", headers,
                   params={"period": "all", "timeframe": "1D"})
    h = json.loads(data)
    df = pd.DataFrame({"date": pd.to_datetime(h["timestamp"], unit="s"), "equity": h["equity"]}).dropna()
    df = df[df["equity"] > 0].copy()
    df["return_pct"] = df["equity"].pct_change() * 100
    return df.dropna()[["date", "return_pct"]]


IBKR_FLEX = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"


def ibkr_flex_trades(token: str, query_id: str, wait: float = 5.0, tries: int = 6) -> pd.DataFrame:
    """Closed trades from an Interactive Brokers Flex Query (Trades section, with realized P/L).
    Set up once in Client Portal: Reports > Flex Queries > Activity; enable Flex Web Service for a token."""
    params = {"t": token, "q": query_id, "v": "3"}
    root = ET.fromstring(_get(f"{IBKR_FLEX}/SendRequest", params=params)[0])
    if root.findtext("Status") != "Success":
        raise ConnectionError(f"IBKR refused the request: {root.findtext('ErrorMessage') or 'unknown error'}")
    params["q"] = root.findtext("ReferenceCode")
    for _ in range(tries):  # IBKR builds the statement asynchronously
        body = _get(f"{IBKR_FLEX}/GetStatement", params=params)[0]
        stmt = ET.fromstring(body)
        if stmt.tag != "FlexStatementResponse":
            break
        if stmt.findtext("ErrorCode") not in ("1019", "1018"):  # 1019 = still generating, 1018 = rate limit
            raise ConnectionError(f"IBKR: {stmt.findtext('ErrorMessage') or 'statement failed'}")
        time.sleep(wait)
    else:
        raise TimeoutError("IBKR is still generating the statement - try again in a minute.")
    rows = []
    for t in stmt.iter("Trade"):
        pnl = float(t.get("fifoPnlRealized") or 0)
        basis = abs(float(t.get("cost") or 0))
        if pnl and basis:  # only closing trades realize P/L
            rows.append({"date": t.get("tradeDate") or t.get("dateTime"), "symbol": t.get("symbol"),
                         "return_pct": pnl / basis * 100, "pnl": pnl})
    if not rows:
        raise ValueError("No closed trades with realized P/L in that Flex Query - add the Trades section "
                         "with 'FIFO P/L Realized' and 'Cost Basis' fields.")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"].str[:8], format="%Y%m%d", errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def mlflow_runs(host: str, experiment_ids: list[str], token: str | None = None, max_runs: int = 5000) -> pd.DataFrame:
    """Every run in an MLflow experiment (self-hosted or Databricks) - one row per run, metrics + params.
    The row count is how many versions were really tried, which is what multiple-testing checks need."""
    host = host.rstrip("/")
    check_public_url(host)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    rows, page = [], None
    while len(rows) < max_runs:
        body = {"experiment_ids": experiment_ids, "max_results": 1000, **({"page_token": page} if page else {})}
        data = json.loads(_request("POST", f"{host}/api/2.0/mlflow/runs/search", headers, json=body,
                                   allow_redirects=False)[0])
        for run in data.get("runs", []):
            row = {"run_id": run["info"]["run_id"], "status": run["info"].get("status"),
                   "start_time": pd.to_datetime(run["info"].get("start_time"), unit="ms")}
            row.update({f"metric.{m['key']}": m["value"] for m in run.get("data", {}).get("metrics", [])})
            row.update({f"param.{q['key']}": q["value"] for q in run.get("data", {}).get("params", [])})
            rows.append(row)
        page = data.get("next_page_token")
        if not page:
            break
    if not rows:
        raise ValueError("That experiment has no runs.")
    return pd.DataFrame(rows)
