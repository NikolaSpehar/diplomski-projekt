from __future__ import annotations

import streamlit as st
from pathlib import Path

from src.cache import (
    get_raw_df,
    get_prepared_df,
    get_policy_df,
    get_view_df,
)
from src.ui_tabs import (
    render_tab_overview,
    render_tab_topn,
    render_tab_drilldown,
    render_tab_heterogeneity,
)

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="5G Network Energy Optimization", layout="wide")
st.title("📡 5G Network Energy Optimization Dashboard")

# -------------------------
# PATHS
# -------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

FILE_MAP = {
    "Weekday": DATA_DIR / "Performance_5G_Weekday.csv",
    "Weekend": DATA_DIR / "Performance_5G_Weekend.csv",
}

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("Dataset, Window & Policy")

dataset = st.sidebar.selectbox("Select dataset", list(FILE_MAP.keys()))

window_mode = st.sidebar.radio(
    "Analytics window",
    ["Selected time-of-day", "Full day"],
    index=0,
)

tod_order = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
tod = st.sidebar.select_slider("Time of day (30-min bins)", options=tod_order, value="12:00")
tod_bin = tod_order.index(tod)

st.sidebar.divider()
st.sidebar.subheader("Economy Mode simulation")

alpha = st.sidebar.slider(
    "Economy saving fraction α (RRU)",
    min_value=0.10,
    max_value=0.80,
    value=0.45,
    step=0.05,
)

st.sidebar.divider()
st.sidebar.subheader("Heuristic controller")

threshold_scope = st.sidebar.radio(
    "Threshold scope (per time-of-day bin)",
    ["Global", "Per-BaseStation", "Per-Cell"],
    index=0,
)

use_hysteresis = st.sidebar.checkbox("Enable hysteresis (reduce jitter)", value=True)

h_sleep = st.sidebar.slider(
    "Hysteresis margin around p30 (sleep boundary) [pp]",
    0.0, 10.0, 2.0, 0.5,
    disabled=not use_hysteresis,
)

h_eco = st.sidebar.slider(
    "Hysteresis margin around p70 (eco boundary) [pp]",
    0.0, 10.0, 2.0, 0.5,
    disabled=not use_hysteresis,
)

debug = st.sidebar.checkbox("Show debug panel", value=False)

path = FILE_MAP[dataset]
path_str = str(path)
st.sidebar.caption(f"Using file: {path}")
st.sidebar.caption(f"Exists: {path.exists()}")

if not path.exists():
    st.error(f"CSV file not found: {path}\n\nPut the file under: {DATA_DIR}")
    st.stop()

# -------------------------
# LOAD + PREPARE + POLICY (CACHED)
# -------------------------
try:
    df_policy = get_policy_df(
        path_str,
        threshold_scope=threshold_scope,
        alpha=float(alpha),
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
    )
except Exception as e:
    st.error("Pipeline failed (cached load/prepare/policy). Diagnostics below.")
    with st.expander("Diagnostics (pipeline failure)", expanded=True):
        st.write("Exception:", repr(e))
        st.write("Path:", path_str)
    st.stop()

df_view = get_view_df(
    path_str,
    threshold_scope=threshold_scope,
    alpha=float(alpha),
    hysteresis_enabled=bool(use_hysteresis),
    h_sleep=float(h_sleep),
    h_eco=float(h_eco),
    window_mode=window_mode,
    tod_bin=int(tod_bin),
)

window_label = f"Selected time-of-day ({tod})" if window_mode == "Selected time-of-day" else "Full day"

if df_view.empty:
    st.warning("No rows available for the selected window after preprocessing/policy application.")
    st.stop()

# -------------------------
# DEBUG
# -------------------------
if debug:
    with st.expander("Debug: raw/prepared/policy (cached)", expanded=True):
        raw = get_raw_df(path_str)
        prepared = get_prepared_df(path_str)
        st.write("Raw shape:", raw.shape)
        st.write("Prepared shape:", prepared.shape)
        st.write("Policy shape:", df_policy.shape)
        st.write("View shape:", df_view.shape)

        st.write("Policy columns:", list(df_policy.columns))
        st.dataframe(df_policy.head(10), use_container_width=True)

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🌍 Network Overview", "🏭 Top-N Base Stations", "🔍 Base Station Drill-Down", "📊 Heterogeneity"]
)

with tab1:
    render_tab_overview(
        df_view=df_view,
        df_policy_full=df_policy,
        tod=tod,
        dataset=dataset,
        window_label=window_label,
        threshold_scope=threshold_scope,
        alpha=float(alpha),
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
    )

with tab2:
    render_tab_topn(
        df_view=df_view,
        tod=tod,
        dataset=dataset,
        window_label=window_label,
        path_str=path_str,
        threshold_scope=threshold_scope,
        alpha=float(alpha),
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
        window_mode=window_mode,
        tod_bin=int(tod_bin),
    )

with tab3:
    render_tab_drilldown(
        df_view=df_view,
        tod=tod,
        dataset=dataset,
        window_label=window_label,
        threshold_scope=threshold_scope,
        alpha=float(alpha),
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
    )

with tab4:
    render_tab_heterogeneity(
        df_view=df_view,
        tod=tod,
        dataset=dataset,
        window_label=window_label,
        threshold_scope=threshold_scope,
        path_str=path_str,
        alpha=float(alpha),
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
        window_mode=window_mode,
        tod_bin=int(tod_bin),
    )
