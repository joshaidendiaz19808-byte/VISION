import re
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, "src")
from vision import verdict

# ── helpers ───────────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())

RETURN_CANDIDATES = [
    "returnpct", "return", "returns", "ret",
    "pnl", "pnlpct", "pl",
    "profit", "profitpct", "gain", "roi",
]

DATE_CANDIDATES = ["date", "time", "datetime", "exitdate", "closetime"]

def detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_map = {_norm(c): c for c in df.columns}
    for key in candidates:
        if key in norm_map:
            return norm_map[key]
    return None

def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.to_numeric(df[c], errors="coerce").notna().sum() > len(df) * 0.5
    ]

def is_dollar_scale(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return len(vals) > 0 and vals.abs().median() > 10

def show_verdict(df: pd.DataFrame, out: dict):
    is_real = "REAL" in out["verdict"]
    accent = "#1D9E75" if is_real else "#E5484D"
    headline = "REAL" if is_real else "FAKE"
    sub = "Worth paper trading" if is_real else "Don't trade this yet"

    st.markdown(
        f"""
        <div style="background:#1A1D24;border:1px solid {accent}33;
                    border-left:4px solid {accent};border-radius:10px;
                    padding:28px 32px;margin:8px 0 24px 0;">
          <div style="display:flex;align-items:baseline;justify-content:space-between;
                      flex-wrap:wrap;gap:12px;">
            <span style="font-size:2.25rem;font-weight:800;color:{accent};
                        letter-spacing:0.02em;">{headline}</span>
            <span style="background:{accent}22;color:{accent};font-weight:700;
                        font-size:0.85rem;padding:6px 14px;border-radius:999px;
                        border:1px solid {accent}55;">{out['score']}</span>
          </div>
          <div style="color:#9CA3AF;font-size:1.05rem;margin-top:4px;">{sub}</div>
          <div style="color:#6B7280;font-size:0.85rem;margin-top:14px;">
            {len(df)} trades analyzed
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not out["reasons"]:
        st.success("Passed every honesty test. Paper trade it before risking real money.")
        return

    st.markdown(
        "<div style='font-weight:700;font-size:1.1rem;margin-bottom:8px;'>What's wrong</div>",
        unsafe_allow_html=True,
    )
    for reason, fix in zip(out["reasons"], out["fixes"]):
        st.markdown(
            f"""
            <div style="background:#1A1D24;border-radius:8px;padding:16px 20px;margin-bottom:12px;">
              <div style="font-size:1rem;color:#FAFAFA;">
                <span style="color:#E5484D;font-weight:700;margin-right:8px;">✗</span>{reason}
              </div>
              <div style="font-size:0.9rem;color:#9CA3AF;margin-top:8px;padding-left:24px;">
                {fix}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── page ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="VISION", page_icon="🔍")
st.markdown(
    """
    <div style="font-size:2.4rem;font-weight:800;letter-spacing:0.03em;
                color:#FAFAFA;margin-bottom:0.1rem;">VISION</div>
    <div style="font-size:1rem;color:#9CA3AF;margin-bottom:1.75rem;">
        Strategy honesty check. Upload your trades, get the truth.
    </div>
    """,
    unsafe_allow_html=True,
)

if "use_sample" not in st.session_state:
    st.session_state.use_sample = False

st.markdown(
    "Upload a CSV of your **closed trades** — one row per finished trade, with a "
    "% return column (e.g. 0.5 = a 0.5% gain). Most brokers and TradingView can export this. "
    "**Tip:** use your trade/performance history, NOT your raw order log. "
    "Need 100+ trades for a meaningful verdict."
)
st.code("date,return_pct\n2024-01-03,0.42\n2024-01-04,-0.18", language="text")
uploaded = st.file_uploader("Upload a CSV — any format, we'll figure it out", type="csv")

if uploaded:
    st.session_state.use_sample = False
elif st.button("▶ Try a sample"):
    st.session_state.use_sample = True

# ── load ──────────────────────────────────────────────────────────────────────

raw = None
source_label = None

if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        source_label = uploaded.name
    except Exception as e:
        st.error(f"Couldn't parse that file: {e}. Make sure it's a plain CSV.")
        st.stop()
elif st.session_state.use_sample:
    try:
        raw = pd.read_csv("trades_sample.csv")
        source_label = "trades_sample.csv"
        st.info("Showing results for the built-in sample file.")
    except FileNotFoundError:
        st.error("Sample file not found. Run the app from the VISION root directory.")
        st.stop()
else:
    st.stop()

raw.columns = [c.strip() for c in raw.columns]

if raw.empty:
    st.error("That CSV is empty. Add some trades and try again.")
    st.stop()

# ── column detection ──────────────────────────────────────────────────────────

num_cols = numeric_columns(raw)
if not num_cols:
    st.error(
        "No numeric columns found. "
        "VISION needs a column of % returns per trade (e.g. 0.5 for a 0.5% gain). "
        "Check that your CSV has numbers in at least one column."
    )
    st.stop()

auto_col = detect_col(raw, RETURN_CANDIDATES) or num_cols[0]
date_col = detect_col(raw, DATE_CANDIDATES)

# Reset the user override whenever the source file changes.
if st.session_state.get("_last_source") != source_label:
    st.session_state.pop("col_override", None)
    st.session_state["_last_source"] = source_label

# Effective column: user override (if still valid) else auto-detected.
effective_col = st.session_state.get("col_override", auto_col)
if effective_col not in num_cols:
    effective_col = auto_col

# ── dollar-amount guard (only hard stop) ─────────────────────────────────────

if is_dollar_scale(raw[effective_col]):
    st.error(
        f"The **'{effective_col}'** column looks like dollar amounts, not % returns. "
        "VISION needs % returns — 0.5 means a 0.5% gain, not $0.50. "
        "Pick your % column below, or convert your data first."
    )
    st.selectbox("Pick your % return column:", num_cols, key="col_override")
    st.stop()

# ── run verdict ───────────────────────────────────────────────────────────────

returns = pd.to_numeric(raw[effective_col], errors="coerce")
dates = raw[date_col] if date_col else pd.Series(range(len(raw)))
df = pd.DataFrame({"date": dates, "return_pct": returns}).dropna(subset=["return_pct"])

if len(df) == 0:
    st.error(
        f"No usable numbers found in '{effective_col}'. "
        "Make sure that column contains numeric values and isn't all blank."
    )
    st.stop()

try:
    out = verdict(df)
except Exception as e:
    st.error(f"Analysis failed: {e}")
    st.stop()

# ── result ────────────────────────────────────────────────────────────────────

st.markdown("---")
show_verdict(df, out)

st.markdown("---")
st.caption(f"Reading returns from: **'{effective_col}'**")
with st.expander("Wrong column? Change it"):
    st.selectbox(
        "Pick a different column:",
        num_cols,
        index=num_cols.index(effective_col),
        key="col_override",
    )
