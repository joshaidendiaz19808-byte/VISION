"""VISION API: every check behind one HTTP server, shared by the web app and the phone apps.

Keys arrive per request and are never stored or logged. Uploaded/downloaded code is parsed, never run,
and deleted after an hour.
"""
import collections
import dataclasses
import hmac
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from checks import ab_test, ai_review, code_audit, connectors, detect, integrity, ml_model, trading, merge  # noqa: E402

MAX_UPLOAD = 300 * 1024 * 1024
MAX_UNZIPPED = 200 * 1024 * 1024
CODE_TTL_S = 3600
CODE_DIRS: dict[str, tuple[Path, float]] = {}

app = FastAPI(title="VISION API", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.state.allow_server_ai_key = os.environ.get("VISION_ALLOW_SERVER_AI_KEY") == "1"
# Access codes you hand out (comma-separated). Unset = open, for running on your own machine.
app.state.access_codes = [c.strip() for c in os.environ.get("VISION_ACCESS_CODES", "").split(",") if c.strip()]
app.state.limits = {"check": (int(os.environ.get("VISION_CHECKS_PER_HOUR", 60)), 3600),
                    "ai": (int(os.environ.get("VISION_AI_REVIEWS_PER_DAY", 10)), 86400),
                    "bad_code": (20, 3600)}  # stops guessing codes
HITS: dict[tuple[str, str], collections.deque] = collections.defaultdict(collections.deque)


def over_limit(who: str, bucket: str) -> int:
    """Sliding-window count per person. Returns seconds until they can go again, or 0. In-memory: one server."""
    n, window = app.state.limits[bucket]
    q, now = HITS[(who, bucket)], time.time()
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= n:
        return int(window - (now - q[0])) + 1
    q.append(now)
    return 0


@app.middleware("http")  # registered before CORS so CORS wraps it and the phone apps can read a 401/429
async def gate(request: Request, call_next):
    if request.method != "POST" or not request.url.path.startswith("/api/"):
        return await call_next(request)
    who = request.client.host if request.client else "?"
    if app.state.access_codes:
        code = request.headers.get("x-vision-code", "")
        if not any(hmac.compare_digest(code.encode(), c.encode()) for c in app.state.access_codes):
            if over_limit(who, "bad_code"):
                return JSONResponse({"detail": "Too many wrong codes - try again later."}, status_code=429)
            return JSONResponse({"detail": "Enter your access code." if not code else "That access code isn't valid."},
                                status_code=401)
        who = "code:" + code
    wait = over_limit(who, "ai" if request.url.path == "/api/ai-review" else "check")
    if wait:
        when = f"{wait // 60 + 1} min" if wait < 3600 else f"{wait // 3600 + 1} h"
        return JSONResponse({"detail": f"Usage limit reached - try again in about {when}."}, status_code=429,
                            headers={"Retry-After": str(wait)})
    return await call_next(request)


# The phone apps load from capacitor://localhost (iOS) / https://localhost (Android).
app.add_middleware(CORSMiddleware, allow_origins=["capacitor://localhost", "https://localhost", "http://localhost"],
                   allow_origin_regex=r"http://localhost:\d+", allow_methods=["*"], allow_headers=["*"])


# ── helpers ───────────────────────────────────────────────────────────────────

def clean(x):
    """Make check output JSON-safe: numpy -> python, NaN/inf -> None."""
    if dataclasses.is_dataclass(x):
        return clean(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return x


async def read_upload(f: UploadFile) -> bytes:
    data = await f.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"{f.filename} is over 300 MB.")
    return data


def parse_table(name: str, data: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    reader = pd.read_excel if name.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    try:
        df, raw = reader(io.BytesIO(data)), reader(io.BytesIO(data), dtype=str)
    except Exception as e:
        raise HTTPException(400, f"Couldn't read {name}: {e}")
    df.columns = raw.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise HTTPException(400, f"{name} is empty.")
    return df, raw


def fetch(fn, *args):
    """Run a connector; its errors are already human-readable and never contain keys."""
    try:
        return fn(*args)
    except (PermissionError, FileNotFoundError, ConnectionError, TimeoutError, ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


def sweep():
    now = time.time()
    for cid, (d, t) in list(CODE_DIRS.items()):
        if now - t > CODE_TTL_S:
            shutil.rmtree(d, ignore_errors=True)
            CODE_DIRS.pop(cid, None)


def opts(raw: str | None) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        raise HTTPException(400, "options must be JSON.")


def numeric_columns(df):
    return [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() > len(df) * 0.5]


def run_table(df: pd.DataFrame, raw: pd.DataFrame, kind: str, o: dict) -> dict:
    """Detect what the table is, pick columns (user choice wins), run the matching checks."""
    guessed, cols = detect.guess(df)
    kind = guessed if kind == "auto" else kind
    allc = list(df.columns)

    def col(key, fallback):
        c = o.get(key) or fallback
        if c is not None and c not in allc:
            raise HTTPException(400, f"No column called '{c}'.")
        return c

    if kind == "trading":
        nums = numeric_columns(df)
        if not nums:
            raise HTTPException(400, "No numeric columns - VISION needs a % return per trade.")
        r = col("returns", cols.get("returns") or detect.find(df, detect.RETURN) or nums[0])
        vals = pd.to_numeric(df[r], errors="coerce").dropna()
        if len(vals) and vals.abs().median() > 10:
            raise HTTPException(400, f"'{r}' looks like dollar amounts, not % returns (0.5 = a 0.5% gain). "
                                     "Pick your % column.")
        d = col("date", cols.get("date") or detect.find(df, detect.DATE))
        ordered = df.sort_values(d) if d else df
        rep = trading.run(ordered[r], cost_bps=float(o.get("cost_bps", 5)), n_trials=int(o.get("n_trials", 1)))
        chosen = {"returns": r, "date": d}
    elif kind == "ab":
        g = col("group", cols.get("group") or detect.find(df, detect.GROUP) or allc[0])
        out = col("outcome", cols.get("outcome") or detect.find(df, detect.OUTCOME) or allc[-1])
        t = col("time", cols.get("time") or detect.find(df, detect.DATE))
        rep = ab_test.run(df, g, out, o.get("control"), t, float(o.get("expected_split", 0.5)),
                          int(o.get("n_metrics", 1)))
        chosen = {"group": g, "outcome": out, "time": t,
                  "groups": sorted(df[g].astype(str).unique().tolist())[:20]}
    elif kind == "predictions":
        yt = col("y_true", cols.get("y_true") or allc[0])
        yp = col("y_pred", cols.get("y_pred") or allc[-1])
        rep = ml_model.run(pred=df, y_true=yt, y_pred=yp)
        chosen = {"y_true": yt, "y_pred": yp}
    elif kind == "data":
        rep = integrity.run(df, raw)
        chosen = {}
    else:
        raise HTTPException(400, f"Unknown kind '{kind}'.")
    return {"kind": kind, "detected": guessed, "columns": allc, "chosen": chosen, "report": clean(rep)}


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/check")
async def check(kind: str = Form("auto"), options: str | None = Form(None), file: UploadFile | None = File(None)):
    """Results from an uploaded CSV/Excel file."""
    if file is None:
        raise HTTPException(400, "Upload a CSV or Excel file.")
    df, raw = parse_table(file.filename or "upload.csv", await read_upload(file))
    return run_table(df, raw, kind, opts(options))


@app.post("/api/connect")
def connect(body: dict):
    """Results pulled from an account. body: {provider, kind, options, credentials:{...}}"""
    p, c = body.get("provider"), body.get("credentials") or {}
    if p == "alpaca":
        df = fetch(connectors.alpaca_returns, c.get("key_id", ""), c.get("secret", ""), bool(c.get("paper", True)))
    elif p == "ibkr":
        df = fetch(connectors.ibkr_flex_trades, c.get("token", ""), c.get("query_id", ""))
    elif p == "url":
        if not str(c.get("url", "")).startswith(("https://", "http://")):
            raise HTTPException(400, "Enter a full http(s) URL.")
        df = fetch(connectors.url_table, c["url"], c.get("header") or None, c.get("key") or None,
                   c.get("json_path") or None)
    else:
        raise HTTPException(400, "provider must be alpaca, ibkr or url.")
    if df is None or df.empty:
        raise HTTPException(400, "That account returned no rows.")
    return run_table(df, df.astype(str), body.get("kind", "auto"), body.get("options") or {})


@app.post("/api/check/ml")
async def check_ml(options: str | None = Form(None), train: UploadFile | None = File(None),
                   test: UploadFile | None = File(None), predictions: UploadFile | None = File(None)):
    o = opts(options)
    args = {}
    if train and test:
        tr = parse_table(train.filename, await read_upload(train))[0]
        te = parse_table(test.filename, await read_upload(test))[0]
        target = o.get("target") or detect.find(tr, detect.YTRUE)
        if target not in tr.columns:
            raise HTTPException(400, "Pick the target column.")
        args.update(train=tr, test=te, target=target, time_col=o.get("time") or detect.find(tr, detect.DATE),
                    id_col=o.get("id"))
    if predictions:
        pr = parse_table(predictions.filename, await read_upload(predictions))[0]
        args.update(pred=pr, y_true=o.get("y_true") or detect.find(pr, detect.YTRUE),
                    y_pred=o.get("y_pred") or detect.find(pr, detect.YPRED))
        if args["y_true"] not in pr.columns or args["y_pred"] not in pr.columns:
            raise HTTPException(400, "Pick the actual and predicted columns.")
    if o.get("run_scores"):
        args.update(run_scores=pd.Series(o["run_scores"]), n_test=int(o.get("n_test", 0)),
                    higher_is_better=bool(o.get("higher_is_better", True)))
    if not args:
        raise HTTPException(400, "Upload train + test tables, and/or a predictions table.")
    rep = ml_model.run(**args)
    columns = {k: list(v.columns) for k, v in (("train", args.get("train")), ("predictions", args.get("pred")))
               if v is not None}
    return {"kind": "ml", "columns": columns, "report": clean(rep)}


@app.post("/api/mlflow")
def mlflow(body: dict):
    """Count every run in an MLflow/Databricks experiment; return metric values for best-of-runs."""
    ids = [e.strip() for e in str(body.get("experiment_ids", "")).split(",") if e.strip()]
    if not body.get("host") or not ids:
        raise HTTPException(400, "Need the tracking server URL and experiment ID(s).")
    runs = fetch(connectors.mlflow_runs, body["host"], ids, body.get("token") or None)
    metrics = {c.removeprefix("metric."): clean(runs[c].dropna().tolist()) for c in runs.columns
               if c.startswith("metric.")}
    return {"runs": len(runs), "metrics": metrics}


@app.post("/api/code")
async def code(files: list[UploadFile] = File(default=[]), repo: str = Form(""), token: str = Form("")):
    """Static audit of uploaded code or a GitHub repo. Returns a code_id for the optional AI review."""
    sweep()
    if files:
        d = Path(tempfile.mkdtemp(prefix="vision_code_"))
        for f in files:
            data = await read_upload(f)
            name = Path(f.filename or "code.py").name
            if name.endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        if sum(i.file_size for i in z.infolist()) > MAX_UNZIPPED:
                            raise HTTPException(413, f"{name} unpacks to more than 200 MB.")
                        z.extractall(d)  # zipfile strips absolute paths and '..'
                except zipfile.BadZipFile:
                    raise HTTPException(400, f"{name} isn't a valid zip.")
            elif name.endswith((".py", ".ipynb")):
                (d / name).write_bytes(data)
            else:
                raise HTTPException(400, f"{name}: upload .py, .ipynb or .zip files.")
    elif repo.strip():
        d = fetch(connectors.github_repo, repo, token or None)
    else:
        raise HTTPException(400, "Upload code files or enter a GitHub repo.")
    cid = uuid.uuid4().hex
    CODE_DIRS[cid] = (d, time.time())
    return {"code_id": cid, "report": clean(code_audit.run(d))}


@app.post("/api/ai-review")
def ai(body: dict):
    sweep()
    entry = CODE_DIRS.get(body.get("code_id", ""))
    if not entry:
        raise HTTPException(404, "That code has expired - audit it again.")
    key = body.get("anthropic_key") or None
    if not key and not app.state.allow_server_ai_key:
        raise HTTPException(400, "Add your Anthropic API key to run the AI review.")
    static = code_audit.run(entry[0])
    rep = ai_review.run(entry[0], key, static)
    return {"summary": rep.get("summary"), "report": clean(rep)}


@app.post("/api/combine")
def combine(body: dict):
    """One verdict over several reports (results + code + AI review), computed by the same rules."""
    from checks.common import Check
    reports = [{"checks": [Check(**{k: c[k] for k in ("name", "status", "reason", "fix", "detail")})
                           for c in r["checks"]]} for r in body.get("reports", [])]
    return clean(merge("Results + the code behind them", reports))


WEB = ROOT / "web"
if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
