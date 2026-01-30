# app.py
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pandas.io.formats.style')

import streamlit as st
from pathlib import Path

from src.cache import (
    get_raw_df,
    get_policy_df,
    get_view_df,
)
from src.ui_tabs import (
    render_tab_overview,
    render_tab_topn,
    render_tab_drilldown,
    render_tab_heterogeneity,
    render_tab_risk_optimization,
    render_tab_distribution_check,
)

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Network Energy Optimization", layout="wide")
st.title("📡 Network Energy Optimization Dashboard")

# -------------------------
# PATHS
# -------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MODELS_DIR = APP_DIR / "models"

FILE_MAP = {
    "5G Weekday (Training)": DATA_DIR / "Performance_5G_Weekday.csv",
    "5G Weekend (Evaluation)": DATA_DIR / "Performance_5G_Weekend.csv",
    "4G Weekday (Inference)": DATA_DIR / "Performance_4G_Weekday.csv",
    "4G Weekend (Inference)": DATA_DIR / "Performance_4G_Weekend.csv",
}

# -------------------------
# SESSION DEFAULTS (one-time)
# -------------------------
def _init_defaults() -> None:
    ss = st.session_state
    ss.setdefault("dataset_label", list(FILE_MAP.keys())[0])
    ss.setdefault("window_mode", "Selected time-of-day")
    ss.setdefault("tod", "12:00")

    ss.setdefault("alpha", 0.45)
    ss.setdefault("controller_type", "ML")

    # Heuristic
    ss.setdefault("threshold_scope", "Global")
    ss.setdefault("use_hysteresis", True)
    ss.setdefault("h_sleep", 2.0)
    ss.setdefault("h_eco", 2.0)

    # ML controller
    ss.setdefault("ml_model_path", str(MODELS_DIR / "sleep_on_5g_weekday.joblib"))
    ss.setdefault("ml_hyst_en", True)
    ss.setdefault("ml_tau_on", 0.80)
    ss.setdefault("ml_tau_off", 0.70)

    # ML features (must match training)
    ss.setdefault("ml_use_energy", False)
    ss.setdefault("ml_use_prev", True)
    ss.setdefault("ml_use_time", True)
    ss.setdefault("ml_use_cyc", True)


_init_defaults()

# -------------------------
# SIDEBAR CONTROLS (FORM => no rerun storm)
# -------------------------
st.sidebar.header("Dataset & Policy")

with st.sidebar.form("controls_form", clear_on_submit=False):
    dataset_label = st.selectbox(
        "Select dataset",
        list(FILE_MAP.keys()),
        index=list(FILE_MAP.keys()).index(st.session_state["dataset_label"]),
        key="dataset_label",
    )
    path = FILE_MAP[dataset_label]
    path_str = str(path)

    if not path.exists():
        st.error(f"File not found: {path}\n\nPlease place the CSV file in: {DATA_DIR}")
        st.stop()

    # --- auto-detect GT based on RAW headers
    raw_preview = get_raw_df(path_str)
    gt_cols = ["ds_ms", "Deep Sleep Time (Millisecond)", "Deep Sleep Time"]
    has_gt = any(c in raw_preview.columns for c in gt_cols)

    if has_gt:
        st.success("✅ Ground Truth detected (5G)")
    else:
        st.info("ℹ️ No Ground Truth (4G Mode). Running inference only.")

    # --- Window
    st.divider()
    st.subheader("Analytics window")

    st.radio(
        "Analytics window",
        ["Selected time-of-day", "Full day"],
        index=0 if st.session_state["window_mode"] == "Selected time-of-day" else 1,
        key="window_mode",
    )

    tod_order = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
    st.select_slider(
        "Time of day (30-min bins)",
        options=tod_order,
        key="tod", 
    )
    tod_bin = tod_order.index(st.session_state["tod"])

    # --- Controller
    st.divider()
    st.subheader("Controller")

    st.slider(
        "Economy saving fraction α (RRU)",
        0.1, 0.9, float(st.session_state["alpha"]), 0.05,
        key="alpha",
    )
    st.radio(
        "Type",
        ["Heuristic", "ML"],
        index=0 if st.session_state["controller_type"] == "Heuristic" else 1,
        key="controller_type",
    )

    # Defaults (will be overridden by form widgets)
    if st.session_state["controller_type"] == "Heuristic":
        st.selectbox(
            "Threshold scope",
            ["Global", "Per-BaseStation", "Per-Cell"],
            index=["Global", "Per-BaseStation", "Per-Cell"].index(st.session_state["threshold_scope"]),
            key="threshold_scope",
        )

        st.checkbox("Hysteresis", value=bool(st.session_state["use_hysteresis"]), key="use_hysteresis")
        if st.session_state["use_hysteresis"]:
            st.slider("Hysteresis (Sleep) [pp]", 0.0, 5.0, float(st.session_state["h_sleep"]), key="h_sleep")
            st.slider("Hysteresis (Eco) [pp]", 0.0, 5.0, float(st.session_state["h_eco"]), key="h_eco")

        # Provide placeholders so downstream code has values
        st.session_state.setdefault("ml_model_path", str(MODELS_DIR / "sleep_on_5g_weekday.joblib"))
    else:
        st.text_input("Model path", value=st.session_state["ml_model_path"], key="ml_model_path")
        st.checkbox("Hysteresis", value=bool(st.session_state["ml_hyst_en"]), key="ml_hyst_en")
        st.slider("Enter ECO (p >= )", 0.0, 1.0, float(st.session_state["ml_tau_on"]), key="ml_tau_on")
        st.slider(
            "Exit ECO (p <= )",
            0.0, 1.0, float(st.session_state["ml_tau_off"]),
            key="ml_tau_off",
            disabled=not bool(st.session_state["ml_hyst_en"]),
        )

        with st.expander("Feature Spec (Advanced)"):
            st.caption("Must match trained model configuration")
            st.checkbox("Energy Features", value=bool(st.session_state["ml_use_energy"]), key="ml_use_energy")
            st.checkbox("Prev Features", value=bool(st.session_state["ml_use_prev"]), key="ml_use_prev")
            st.checkbox("Time Features", value=bool(st.session_state["ml_use_time"]), key="ml_use_time")
            st.checkbox("Cyclical Time", value=bool(st.session_state["ml_use_cyc"]), key="ml_use_cyc")

    apply_clicked = st.form_submit_button("Apply")

# If user hasn't clicked Apply yet in this session, we still proceed using defaults.
# (Form prevents rerun storms while dragging; only commits on click.)
# Streamlit will keep session_state values anyway.
window_mode = st.session_state["window_mode"]
tod = st.session_state["tod"]
tod_bin = tod_order.index(tod)
controller_type = st.session_state["controller_type"]
alpha = float(st.session_state["alpha"])

# Heuristic params
threshold_scope = st.session_state.get("threshold_scope", "Global")
use_hysteresis = bool(st.session_state.get("use_hysteresis", True))
h_sleep = float(st.session_state.get("h_sleep", 2.0))
h_eco = float(st.session_state.get("h_eco", 2.0))

# ML params
ml_model_path = str(st.session_state.get("ml_model_path", str(MODELS_DIR / "sleep_on_5g_weekday.joblib")))
ml_hyst_en = bool(st.session_state.get("ml_hyst_en", True))
ml_tau_on = float(st.session_state.get("ml_tau_on", 0.80))
ml_tau_off = float(st.session_state.get("ml_tau_off", 0.70))

ml_use_energy = bool(st.session_state.get("ml_use_energy", False))
ml_use_prev = bool(st.session_state.get("ml_use_prev", True))
ml_use_time = bool(st.session_state.get("ml_use_time", True))
ml_use_cyc = bool(st.session_state.get("ml_use_cyc", True))

# -------------------------
# COMPUTATION
# -------------------------
try:
    df_policy = get_policy_df(
        path_str,
        controller_type=controller_type,
        alpha=float(alpha),
        threshold_scope=threshold_scope,
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
        ml_model_path=ml_model_path if controller_type == "ML" else "",
        ml_tau_on=float(ml_tau_on),
        ml_tau_off=float(ml_tau_off),
        ml_hysteresis_enabled=bool(ml_hyst_en),
        ml_use_energy_features=bool(ml_use_energy),
        ml_use_prev_features=bool(ml_use_prev),
        ml_use_time_features=bool(ml_use_time),
        ml_use_time_cyclical=bool(ml_use_cyc),
    )

    df_view = get_view_df(
        path_str,
        controller_type=controller_type,
        alpha=float(alpha),
        threshold_scope=threshold_scope,
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
        ml_model_path=ml_model_path if controller_type == "ML" else "",
        ml_tau_on=float(ml_tau_on),
        ml_tau_off=float(ml_tau_off),
        ml_hysteresis_enabled=bool(ml_hyst_en),
        ml_use_energy_features=bool(ml_use_energy),
        ml_use_prev_features=bool(ml_use_prev),
        ml_use_time_features=bool(ml_use_time),
        ml_use_time_cyclical=bool(ml_use_cyc),
        window_mode=window_mode,
        tod_bin=int(tod_bin),
    )
except Exception as e:
    st.error(f"Pipeline computation failed. Details: {e}")
    st.stop()

window_label = f"{window_mode} ({tod})" if "Selected" in window_mode else "Full Day"

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🌍 Overview", "🏭 Top-N Savings", "🔍 Drill-Down", "📊 Heterogeneity", "⚖️ Risk & Optimization", "📉 Drift Detection"]
)

with tab1:
    render_tab_overview(
        df_view=df_view,
        df_policy_full=df_policy,
        tod=tod,
        dataset=dataset_label,
        window_label=window_label,
        controller_type=controller_type,
        threshold_scope=threshold_scope,
        alpha=float(alpha),
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
        ml_tau_on=float(ml_tau_on),
        ml_tau_off=float(ml_tau_off),
        ml_hysteresis_enabled=bool(ml_hyst_en),
        show_gt_metrics=has_gt,
    )

with tab2:
    render_tab_topn(
        df_view=df_view,
        tod=tod,
        dataset=dataset_label,
        window_label=window_label,
        path_str=path_str,
        controller_type=controller_type,
        threshold_scope=threshold_scope,
        alpha=float(alpha),
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
        ml_model_path=ml_model_path if controller_type == "ML" else "",
        ml_tau_on=float(ml_tau_on),
        ml_tau_off=float(ml_tau_off),
        ml_hysteresis_enabled=bool(ml_hyst_en),
        ml_use_energy_features=bool(ml_use_energy),
        ml_use_prev_features=bool(ml_use_prev),
        ml_use_time_features=bool(ml_use_time),
        ml_use_time_cyclical=bool(ml_use_cyc),
        window_mode=window_mode,
        tod_bin=int(tod_bin),
        show_gt_metrics=has_gt,
    )

with tab3:
    render_tab_drilldown(
        df_view=df_view,
        tod=tod,
        dataset=dataset_label,
        window_label=window_label,
        controller_type=controller_type,
        threshold_scope=threshold_scope,
        alpha=float(alpha),
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
        show_gt_metrics=has_gt,
    )

with tab4:
    render_tab_heterogeneity(
        df_view=df_view,
        tod=tod,
        dataset=dataset_label,
        window_label=window_label,
        controller_type=controller_type,
        threshold_scope=threshold_scope,
        path_str=path_str,
        alpha=float(alpha),
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
        ml_model_path=ml_model_path if controller_type == "ML" else "",
        ml_tau_on=float(ml_tau_on),
        ml_tau_off=float(ml_tau_off),
        ml_hysteresis_enabled=bool(ml_hyst_en),
        ml_use_energy_features=bool(ml_use_energy),
        ml_use_prev_features=bool(ml_use_prev),
        ml_use_time_features=bool(ml_use_time),
        ml_use_time_cyclical=bool(ml_use_cyc),
        window_mode=window_mode,
        tod_bin=int(tod_bin),
        show_gt_metrics=has_gt,
    )

with tab5:
    if controller_type != "ML":
        st.warning("Optimization is only available for the ML Controller.")
    else:
        render_tab_risk_optimization(
            df_view=df_view,
            df_policy_full=df_policy,
            alpha=float(alpha),
            current_tau_on=float(ml_tau_on),
            current_tau_off=float(ml_tau_off),
        )

with tab6:
    render_tab_distribution_check(
        df_view=df_view,
        path_5g_train=str(FILE_MAP["5G Weekday (Training)"]),
    )
