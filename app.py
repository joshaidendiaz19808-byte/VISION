import math
import os
import re
import sys
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from vision import verdict
import abtest

# ── helpers ───────────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

RETURN_CANDIDATES = [
    "returnpct", "return", "returns", "ret",
    "pnl", "pnlpct", "pl",
    "profit", "profitpct", "gain", "roi",
]
DATE_CANDIDATES = ["date", "time", "datetime", "timestamp", "day", "exitdate", "closetime"]
VARIANT_CANDIDATES = ["variant", "group", "version", "arm", "bucket", "testgroup", "experimentgroup", "cohort"]
METRIC_CANDIDATES = [
    "converted", "conversion", "purchased", "purchase", "signup", "signedup", "clicked", "click",
    "revenue", "value", "amount", "orders", "result", "outcome", "metric",
]

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

def sample_path(name: str) -> str:
    return os.path.join(HERE, name)

def show_verdict(n_label: str, out: dict, is_real: bool, headline: str, sub: str):
    accent = "#1D9E75" if is_real else "#E5484D"
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
          <div style="color:#6B7280;font-size:0.85rem;margin-top:14px;">{n_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def show_problems(out: dict, all_clear: str):
    if not out["reasons"]:
        st.success(all_clear)
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

def load_csv(uploaded, sample_key: str, sample_file: str):
    """Returns (DataFrame, label) or stops the page if nothing to show yet."""
    if sample_key not in st.session_state:
        st.session_state[sample_key] = False
    if uploaded:
        st.session_state[sample_key] = False
    elif st.button("▶ Try a sample", key=f"btn_{sample_key}"):
        st.session_state[sample_key] = True

    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't parse that file: {e}. Make sure it's a plain CSV.")
            st.stop()
        label = uploaded.name
    elif st.session_state[sample_key]:
        try:
            raw = pd.read_csv(sample_path(sample_file))
        except FileNotFoundError:
            st.error("Sample file not found.")
            st.stop()
        label = sample_file
        st.info("Showing results for the built-in sample file.")
    else:
        st.stop()

    raw.columns = [str(c).strip() for c in raw.columns]
    if raw.empty:
        st.error("That CSV is empty. Add some rows and try again.")
        st.stop()
    return raw, label

# ── mode 1: trading strategy ─────────────────────────────────────────────────

def run_strategy():
    st.markdown(
        "Upload a CSV of your **closed trades** — one row per finished trade, with a "
        "% return column (e.g. 0.5 = a 0.5% gain). Most brokers and TradingView can export this. "
        "**Tip:** use your trade/performance history, NOT your raw order log. "
        "Need 100+ trades for a meaningful verdict."
    )
    st.code("date,return_pct\n2024-01-03,0.42\n2024-01-04,-0.18", language="text")
    uploaded = st.file_uploader("Upload a CSV — any format, we'll figure it out", type="csv", key="strat_up")
    raw, source_label = load_csv(uploaded, "use_sample", "trades_sample.csv")

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

    effective_col = st.session_state.get("col_override", auto_col)
    if effective_col not in num_cols:
        effective_col = auto_col

    if is_dollar_scale(raw[effective_col]):
        st.error(
            f"The **'{effective_col}'** column looks like dollar amounts, not % returns. "
            "VISION needs % returns — 0.5 means a 0.5% gain, not $0.50. "
            "Pick your % column below, or convert your data first."
        )
        st.selectbox("Pick your % return column:", num_cols, key="col_override")
        st.stop()

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

    st.markdown("---")
    is_real = "REAL" in out["verdict"]
    show_verdict(
        f"{len(df)} trades analyzed", out, is_real,
        "REAL" if is_real else "FAKE",
        "Worth paper trading" if is_real else "Don't trade this yet",
    )
    show_problems(out, "Passed every honesty test. Paper trade it before risking real money.")

    st.markdown("---")
    st.caption(f"Reading returns from: **'{effective_col}'**")
    with st.expander("Wrong column? Change it"):
        st.selectbox(
            "Pick a different column:",
            num_cols,
            index=num_cols.index(effective_col),
            key="col_override",
        )

# ── mode 2: A/B test ──────────────────────────────────────────────────────────

def show_ab_result(out: dict):
    s = out["summary"]
    st.markdown("---")
    if out["is_real"]:
        winner = s["label_b"] if s["diff"] > 0 else s["label_a"]
        headline, sub = "REAL WIN", f"{winner} is better. Safe to roll out."
    else:
        headline, sub = "NOT PROVEN", "Don't roll this out yet - the win could be fake."
    show_verdict(f"{s['n_a'] + s['n_b']:,} users analyzed", out, out["is_real"], headline, sub)
    if s["kind"] == "binary":
        fmt = lambda v: f"{v * 100:.2f}%"
        diff_fmt = lambda v: f"{v * 100:+.2f} pts"
    else:
        fmt = lambda v: f"{v:,.2f}"
        diff_fmt = lambda v: f"{v:+,.2f}"
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{s['label_a']} (original)", fmt(s["value_a"]), help=f"{s['n_a']:,} users")
    c2.metric(f"{s['label_b']} (new)", fmt(s["value_b"]),
              None if math.isnan(s["lift"]) else f"{s['lift'] * 100:+.1f}%",
              help=f"{s['n_b']:,} users")
    c3.metric("Chance it's luck", f"{s['p_value'] * 100:.1f}%", help="p-value: under 5% is the usual bar.")
    st.caption(
        f"{s['metric'].capitalize()}: 95% sure the true difference ({s['label_b']} minus {s['label_a']}) "
        f"is between {diff_fmt(s['ci_low'])} and {diff_fmt(s['ci_high'])}."
    )
    show_problems(out, "Passed every honesty test. Roll it out — and keep an eye on the numbers for a few weeks.")

def run_ab_counts():
    st.markdown("Just have the totals from your testing tool? Type them in.")
    a, b = st.columns(2)
    with a:
        st.markdown("**A — original**")
        va = st.number_input("Visitors", min_value=0, value=0, step=1, key="va")
        ca = st.number_input("Conversions (sign-ups, sales, clicks…)", min_value=0, value=0, step=1, key="ca")
    with b:
        st.markdown("**B — new version**")
        vb = st.number_input("Visitors", min_value=0, value=0, step=1, key="vb")
        cb = st.number_input("Conversions (sign-ups, sales, clicks…)", min_value=0, value=0, step=1, key="cb")
    if min(va, vb) == 0:
        st.stop()
    if ca > va or cb > vb:
        st.error("Conversions can't be more than visitors.")
        st.stop()
    show_ab_result(abtest.check_counts(va, ca, vb, cb))

def run_ab_csv():
    st.markdown(
        "Upload a CSV with **one row per visitor/user**: which version they saw, and what happened "
        "(converted yes/no, or a number like revenue). Add a date column and we'll also check the win "
        "holds up over time."
    )
    st.code("date,variant,converted\n2026-03-01,A,0\n2026-03-01,B,1", language="text")
    uploaded = st.file_uploader("Upload a CSV", type="csv", key="ab_up")
    raw, label = load_csv(uploaded, "ab_sample", "ab_sample.csv")

    cols = list(raw.columns)
    two_valued = [c for c in cols if raw[c].nunique(dropna=True) == 2]
    num_cols = numeric_columns(raw)
    auto_variant = detect_col(raw, VARIANT_CANDIDATES) or (two_valued[0] if two_valued else cols[0])
    auto_metric = detect_col(raw, METRIC_CANDIDATES) or next((c for c in num_cols if c != auto_variant), cols[-1])
    auto_date = detect_col(raw, DATE_CANDIDATES)

    if st.session_state.get("_ab_source") != label:
        for k in ("ab_variant", "ab_metric", "ab_date"):
            st.session_state.pop(k, None)
        st.session_state["_ab_source"] = label

    variant_col = st.session_state.get("ab_variant", auto_variant)
    metric_col = st.session_state.get("ab_metric", auto_metric)
    date_opt = st.session_state.get("ab_date", auto_date or "(none)")
    date_col = None if date_opt == "(none)" else date_opt

    try:
        out = abtest.check_rows(raw, variant_col, metric_col, date_col)
    except Exception as e:
        st.error(f"Couldn't analyze that: {e}")
        out = None
    if out:
        show_ab_result(out)

    st.markdown("---")
    st.caption(f"Version column: **'{variant_col}'** · Result column: **'{metric_col}'** · "
               f"Date column: **'{date_col or 'none'}'**")
    with st.expander("Wrong columns? Change them", expanded=out is None):
        st.selectbox("Which column says which version they saw?", cols,
                     index=cols.index(variant_col), key="ab_variant")
        st.selectbox("Which column is the result?", cols, index=cols.index(metric_col), key="ab_metric")
        date_opts = ["(none)"] + cols
        st.selectbox("Date column (optional)", date_opts,
                     index=date_opts.index(date_opt) if date_opt in date_opts else 0, key="ab_date")

def run_ab():
    how = st.radio("What do you have?", ["A CSV of results", "Just the totals"], horizontal=True)
    if how == "Just the totals":
        run_ab_counts()
    else:
        run_ab_csv()

# ── page ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="VISION", page_icon="🔍")
st.markdown(
    """
    <div style="font-size:2.4rem;font-weight:800;letter-spacing:0.03em;
                color:#FAFAFA;margin-bottom:0.1rem;">VISION</div>
    <div style="font-size:1rem;color:#9CA3AF;margin-bottom:1.25rem;">
        Is it real, or just luck? Upload your results, get the truth.
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.radio("What are you checking?", ["Trading strategy", "A/B test"], horizontal=True)
st.markdown("")
if mode == "Trading strategy":
    run_strategy()
else:
    run_ab()
