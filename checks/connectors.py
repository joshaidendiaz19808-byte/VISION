"""Pull results straight from where people keep them, using their own API access.

Keys are passed in per call and only ever live in memory: nothing here writes
a key to disk, a log or an error message.
"""
import io
import json
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
import requests

TIMEOUT = 60
MAX_DOWNLOAD = 300 * 1024 * 1024


def _get(url, headers=None, **kw):
    r = requests.get(url, headers=headers or {}, timeout=TIMEOUT, stream=True, **kw)
    if r.status_code in (401, 403):
        raise PermissionError(f"Access denied by {url.split('/')[2]} - check the key has read permission.")
    if r.status_code == 404:
        raise FileNotFoundError(f"Not found: {url} (private? then add a key)")
    r.raise_for_status()
    buf = io.BytesIO()
    for chunk in r.iter_content(1 << 20):
        buf.write(chunk)
        if buf.tell() > MAX_DOWNLOAD:
            raise ValueError("Download is over 300 MB - point VISION at a smaller export.")
    return buf.getvalue(), r.headers.get("content-type", "")


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
    headers = {auth_header: auth_value} if auth_header and auth_value else {}
    data, ctype = _get(url, headers)
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
