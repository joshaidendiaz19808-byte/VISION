"""Look at an uploaded table and guess what kind of result it is, so people don't have to say."""
import re

import pandas as pd


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", c.lower())


def find(df: pd.DataFrame, names) -> str | None:
    cols = {_norm(c): c for c in df.columns}
    return next((cols[n] for n in names if n in cols), None)


RETURN = ["returnpct", "return", "returns", "ret", "pnlpct", "pnl", "pl", "profitpct", "profit", "gain", "roi"]
DATE = ["date", "time", "datetime", "timestamp", "exitdate", "closetime", "createdat"]
GROUP = ["group", "variant", "arm", "bucket", "treatment", "experimentgroup", "cohort"]
OUTCOME = ["converted", "conversion", "outcome", "clicked", "purchased", "revenue", "value", "metric", "success"]
YTRUE = ["ytrue", "actual", "label", "truth", "target", "y"]
YPRED = ["ypred", "pred", "prediction", "predicted", "score", "prob", "probability", "yhat"]


def guess(df: pd.DataFrame) -> tuple[str, dict]:
    """Return (kind, columns) where kind is trading | ab | predictions | data."""
    g, o = find(df, GROUP), find(df, OUTCOME)
    if g and o and 2 <= df[g].nunique() <= 5:
        return "ab", {"group": g, "outcome": o, "time": find(df, DATE)}
    t, p = find(df, YTRUE), find(df, YPRED)
    if t and p and t != p:
        return "predictions", {"y_true": t, "y_pred": p}
    r = find(df, RETURN)
    if r and pd.to_numeric(df[r], errors="coerce").notna().mean() > 0.9:
        return "trading", {"returns": r, "date": find(df, DATE)}
    return "data", {}
