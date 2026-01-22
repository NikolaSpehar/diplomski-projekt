from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import resolve_col
from src.aggregations import bs_summary, bs_mean_prb, state_distribution
from src.kpis import bs_level_gt_vs_pred
from src.policy import compute_thresholds, decide_state
from src import plots

from src.cache import get_bs_summary_cached, get_per_cell_kpis_cached


def _to_kwh(x_wh: float) -> float:
    return float(x_wh) / 1000.0


def render_tab_overview(
    df_view: pd.DataFrame,
    df_policy_full: pd.DataFrame,
    tod: str,
    dataset: str,
    window_label: str,
    threshold_scope: str,
    alpha: float,
    hysteresis_enabled: bool,
    h_sleep: float,
    h_eco: float,
) -> None:
    st.subheader(f"Global Network Overview ({window_label})")

    bs_col = resolve_col(df_view, "bs")
    cell_col = resolve_col(df_view, "cell")
    traffic_col = resolve_col(df_view, "traffic_kb")
    prb_col = resolve_col(df_view, "prb")

    total_bs = int(df_view[bs_col].nunique())
    total_cells = int(df_view[cell_col].nunique())
    total_traffic_kbyte = float(pd.to_numeric(df_view[traffic_col], errors="coerce").sum(skipna=True))

    total_baseline_kwh = _to_kwh(float(pd.to_numeric(df_view["baseline_Wh"], errors="coerce").sum(skipna=True)))
    total_saved_kwh = _to_kwh(float(pd.to_numeric(df_view["eco_saved_Wh"], errors="coerce").sum(skipna=True)))
    total_saved_pct = (100.0 * total_saved_kwh / total_baseline_kwh) if total_baseline_kwh > 0 else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Base Stations", total_bs)
    c2.metric("Cells", total_cells)
    c3.metric("Total Traffic (KByte)", f"{total_traffic_kbyte:,.0f}")
    c4.metric("Baseline energy (kWh)", f"{total_baseline_kwh:,.3f}")
    c5.metric("Eco saved (kWh)", f"{total_saved_kwh:,.3f} ({total_saved_pct:.2f}%)")

    # --- BS mean PRB
    bs_mean = bs_mean_prb(df_view)

    # --- Thresholds for interpretability on BS mean
    if threshold_scope == "Global":
        p30 = pd.to_numeric(df_view["p30"], errors="coerce").dropna() if "p30" in df_view.columns else pd.Series(dtype=float)
        p70 = pd.to_numeric(df_view["p70"], errors="coerce").dropna() if "p70" in df_view.columns else pd.Series(dtype=float)
        bs_mean["p30"] = float(p30.iloc[0]) if len(p30) else float("nan")
        bs_mean["p70"] = float(p70.iloc[0]) if len(p70) else float("nan")
    elif threshold_scope == "Per-BaseStation":
        thr_bs = df_view.groupby(bs_col, dropna=False)[["p30", "p70"]].first().reset_index()
        bs_mean = bs_mean.merge(thr_bs, on=bs_col, how="left")
    else:
        p30g, p70g = compute_thresholds(df_view[prb_col])
        bs_mean["p30"] = p30g
        bs_mean["p70"] = p70g

    bs_mean["State"] = bs_mean.apply(
        lambda r: decide_state(r["mean_prb"], r.get("p30", np.nan), r.get("p70", np.nan)),
        axis=1,
    ).astype(str)

    dist = state_distribution(bs_mean["State"])
    st.altair_chart(
        plots.bar_state_distribution(dist, title="Base Station State Distribution (heuristic)"),
        use_container_width=True,
    )

    gt_metrics = bs_level_gt_vs_pred(df_view, state_col="State") if "State" in df_view.columns else {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TP (SLEEP & GT sleep)", gt_metrics["tp"])
    m2.metric("FP (SLEEP & no GT)", gt_metrics["fp"])
    m3.metric("FN (not SLEEP & GT)", gt_metrics["fn"])
    m4.metric("TN (not SLEEP & no GT)", gt_metrics["tn"])

    sleep_by_bs = df_view.groupby(bs_col, dropna=False)["sleep_on"].mean().reset_index().rename(columns={"sleep_on": "p_sleep_cells"})
    merged = bs_mean.merge(sleep_by_bs, on=bs_col, how="left")

    st.altair_chart(
        plots.scatter_load_vs_sleep(merged, title="Load vs GT deep sleep (per Base Station)"),
        use_container_width=True,
    )

    st.markdown("**Summary (JSON)**")
    st.json(
        {
            "dataset": dataset,
            "time_of_day": tod,
            "window": window_label,
            "base_stations": total_bs,
            "cells": total_cells,
            "total_traffic_kbyte": total_traffic_kbyte,
            "energy_kwh": {
                "baseline_kwh": total_baseline_kwh,
                "eco_saved_kwh": total_saved_kwh,
                "eco_saved_pct": total_saved_pct,
            },
            "policy": {
                "threshold_scope": threshold_scope,
                "alpha": float(alpha),
                "hysteresis_enabled": bool(hysteresis_enabled),
                "h_sleep": float(h_sleep),
                "h_eco": float(h_eco),
            },
            "heuristic_vs_gt_bs_level": gt_metrics,
            "bs_state_distribution_percent": {row["State"]: float(row["Percent"]) for _, row in dist.iterrows()},
        }
    )


def render_tab_topn(
    df_view: pd.DataFrame,
    tod: str,
    dataset: str,
    window_label: str,
    *,
    path_str: str,
    threshold_scope: str,
    alpha: float,
    hysteresis_enabled: bool,
    h_sleep: float,
    h_eco: float,
    window_mode: str,
    tod_bin: int,
) -> None:
    st.subheader(f"Top-N Base Stations ({window_label})")

    summary = get_bs_summary_cached(
        path_str,
        threshold_scope=threshold_scope,
        alpha=alpha,
        hysteresis_enabled=hysteresis_enabled,
        h_sleep=h_sleep,
        h_eco=h_eco,
        window_mode=window_mode,
        tod_bin=tod_bin,
    )

    bs_col = resolve_col(summary, "bs")

    summary = summary.copy()
    summary["baseline_kWh"] = summary["baseline_Wh"].map(_to_kwh)
    summary["eco_saved_kWh"] = summary["eco_saved_Wh"].map(_to_kwh)

    c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
    with c1:
        top_n = st.slider("Select N", 5, 50, 10)
    with c2:
        rank_by = st.selectbox(
            "Rank by",
            ["eco_saved_kWh", "eco_saved_pct", "baseline_kWh", "traffic_kbyte", "p_sleep"],
            index=0,
        )
    with c3:
        ascending = st.checkbox("Ascending", value=False)

    top = summary.sort_values(rank_by, ascending=ascending).head(top_n).copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))

    st.markdown("### A) Ranked KPI table (Top-N)")
    display_cols = [
        "rank",
        bs_col,
        "traffic_kbyte",
        "mean_prb",
        "baseline_kWh",
        "eco_saved_kWh",
        "eco_saved_pct",
        "p_sleep",
    ]
    top_disp = top[display_cols].rename(
        columns={
            bs_col: "Base Station ID",
            "traffic_kbyte": "traffic (KByte)",
            "mean_prb": "mean PRB (%)",
            "baseline_kWh": "baseline energy (kWh)",
            "eco_saved_kWh": "eco saved (kWh)",
            "eco_saved_pct": "eco saved (%)",
            "p_sleep": "GT sleep fraction",
        }
    )

    styler = (
        top_disp.style.format(
            {
                "traffic (KByte)": "{:,.0f}",
                "mean PRB (%)": "{:.2f}",
                "baseline energy (kWh)": "{:,.3f}",
                "eco saved (kWh)": "{:,.3f}",
                "eco saved (%)": "{:.2f}",
                "GT sleep fraction": "{:.3f}",
            }
        )
        .bar(subset=["eco saved (kWh)"], align="zero")
        .bar(subset=["eco saved (%)"], vmin=0, vmax=max(1e-9, float(top_disp["eco saved (%)"].max())))
    )
    st.dataframe(styler, use_container_width=True)

    st.markdown("### B) Impact–feasibility scatter (Top-N)")
    plot_top = top.rename(columns={bs_col: "Base Station ID"}).copy()
    st.altair_chart(
        plots.topn_scatter(plot_top, title="Top-N: savings impact vs savings intensity"),
        use_container_width=True,
    )

    st.markdown("**Summary (JSON)**")
    st.json(
        {
            "dataset": dataset,
            "time_of_day": tod,
            "window": window_label,
            "top_n": int(top_n),
            "rank_by": rank_by,
            "ascending": bool(ascending),
        }
    )


def render_tab_drilldown(
    df_view: pd.DataFrame,
    tod: str,
    dataset: str,
    window_label: str,
    threshold_scope: str,
    alpha: float,
    hysteresis_enabled: bool,
    h_sleep: float,
    h_eco: float,
) -> None:
    from src.kpis import per_cell_kpis

    st.subheader(f"Base Station Drill-Down ({window_label})")

    bs_col = resolve_col(df_view, "bs")
    traffic_col = resolve_col(df_view, "traffic_kb")

    top_bs_for_selector = (
        df_view.groupby(bs_col, dropna=False)[traffic_col]
        .sum()
        .nlargest(25)
        .reset_index()
    )
    if top_bs_for_selector.empty:
        st.info("No base stations available for this window.")
        return

    selected_bs = st.selectbox("Select Base Station", top_bs_for_selector[bs_col])
    df_bs = df_view.loc[df_view[bs_col] == selected_bs].copy()
    if df_bs.empty:
        st.info("No data for the selected base station in this window.")
        return

    cell_kpi = per_cell_kpis(df_bs).sort_values("eco_saved_Wh", ascending=False).reset_index(drop=True)
    cell_kpi = cell_kpi.copy()
    cell_kpi["baseline_kWh"] = cell_kpi["baseline_Wh"].map(_to_kwh)
    cell_kpi["eco_saved_kWh"] = cell_kpi["eco_saved_Wh"].map(_to_kwh)

    st.markdown("### Per-cell KPIs")
    st.dataframe(cell_kpi, use_container_width=True)

    left, right = st.columns(2)

    with left:
        gt_sleep = cell_kpi[["Cell ID", "p_sleep", "f_sleep"]].copy()
        gt_sleep = gt_sleep.rename(columns={"p_sleep": "P(sleep_on)", "f_sleep": "Sleep time fraction"})
        gt_melt = gt_sleep.melt(id_vars=["Cell ID"], var_name="Metric", value_name="Value")
        chart = plots.cell_gt_sleep_bar(gt_melt, title="Ground-truth deep sleep metrics by cell")
        st.altair_chart(chart.properties(height=min(520, 22 * len(gt_sleep) + 120)), use_container_width=True)

    with right:
        sav = cell_kpi[["Cell ID", "eco_saved_pct_of_baseline", "eco_saved_kWh"]].copy()
        sav = sav.sort_values("eco_saved_kWh", ascending=False)
        chart = plots.cell_eco_savings_bar(sav, title="Economy Mode savings by cell")
        st.altair_chart(chart.properties(height=min(520, 22 * len(sav) + 120)), use_container_width=True)

    baseline_total_kwh = float(pd.to_numeric(df_bs["baseline_Wh"], errors="coerce").sum(skipna=True)) / 1000.0
    eco_saved_total_kwh = float(pd.to_numeric(df_bs["eco_saved_Wh"], errors="coerce").sum(skipna=True)) / 1000.0
    eco_saved_pct = (100.0 * eco_saved_total_kwh / baseline_total_kwh) if baseline_total_kwh > 0 else 0.0

    st.markdown("### Selected Base Station Summary")
    s1, s2, s3 = st.columns(3)
    s1.metric("Baseline energy (kWh)", f"{baseline_total_kwh:,.3f}")
    s2.metric("Eco saved (kWh)", f"{eco_saved_total_kwh:,.3f}")
    s3.metric("Eco saved (% baseline)", f"{eco_saved_pct:.2f}%")

    st.markdown("**Summary (JSON)**")
    st.json(
        {
            "dataset": dataset,
            "time_of_day": tod,
            "window": window_label,
            "selected_base_station_id": int(selected_bs) if pd.notna(selected_bs) else None,
            "policy": {
                "threshold_scope": threshold_scope,
                "alpha": float(alpha),
                "hysteresis_enabled": bool(hysteresis_enabled),
                "h_sleep": float(h_sleep),
                "h_eco": float(h_eco),
            },
            "economy_simulation_selected_bs": {
                "baseline_kwh": baseline_total_kwh,
                "eco_saved_kwh": eco_saved_total_kwh,
                "eco_saved_pct": eco_saved_pct,
            },
        }
    )


def render_tab_heterogeneity(
    df_view: pd.DataFrame,
    tod: str,
    dataset: str,
    window_label: str,
    threshold_scope: str,
    *,
    path_str: str,
    alpha: float,
    hysteresis_enabled: bool,
    h_sleep: float,
    h_eco: float,
    window_mode: str,
    tod_bin: int,
) -> None:
    st.subheader(f"Heterogeneity ({window_label})")

    st.caption("Tip: This tab can be expensive. Use the button to compute cached heterogeneity KPIs.")
    compute = st.button("Compute heterogeneity KPIs (cached)", type="primary")
    if not compute:
        st.info("Click the button to compute per-cell and per-base-station heterogeneity KPIs.")
        return

    cell_kpi = get_per_cell_kpis_cached(
        path_str,
        threshold_scope=threshold_scope,
        alpha=alpha,
        hysteresis_enabled=hysteresis_enabled,
        h_sleep=h_sleep,
        h_eco=h_eco,
        window_mode=window_mode,
        tod_bin=tod_bin,
    )
    if cell_kpi.empty:
        st.info("No per-cell KPIs available for this window.")
        return

    cell_kpi = cell_kpi.copy()
    cell_kpi["baseline_kWh"] = cell_kpi["baseline_Wh"].map(_to_kwh)
    cell_kpi["eco_saved_kWh"] = cell_kpi["eco_saved_Wh"].map(_to_kwh)

    bs_kpi = get_bs_summary_cached(
        path_str,
        threshold_scope=threshold_scope,
        alpha=alpha,
        hysteresis_enabled=hysteresis_enabled,
        h_sleep=h_sleep,
        h_eco=h_eco,
        window_mode=window_mode,
        tod_bin=tod_bin,
    )
    bs_col = resolve_col(bs_kpi, "bs")
    bs_kpi = bs_kpi.copy()
    bs_kpi["baseline_kWh"] = bs_kpi["baseline_Wh"].map(_to_kwh)
    bs_kpi["eco_saved_kWh"] = bs_kpi["eco_saved_Wh"].map(_to_kwh)

    def _cv(series: pd.Series) -> float:
        s = pd.to_numeric(series, errors="coerce").dropna()
        m = float(s.mean()) if len(s) else 0.0
        sd = float(s.std(ddof=0)) if len(s) else 0.0
        return float(sd / m) if m > 0 else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cells", int(len(cell_kpi)))
    c2.metric("Base Stations", int(len(bs_kpi)))
    c3.metric("CV(p_sleep) across cells", f"{_cv(cell_kpi['p_sleep']):.3f}")
    c4.metric("CV(eco_saved_kWh) across BS", f"{_cv(bs_kpi['eco_saved_kWh']):.3f}")

    st.markdown("### A) Per-cell heterogeneity")
    a1, a2 = st.columns(2)
    with a1:
        st.altair_chart(
            plots.hist_numeric(cell_kpi, "p_sleep", "Cells: distribution of P(sleep_on)", bin_step=0.05),
            use_container_width=True,
        )
    with a2:
        st.altair_chart(
            plots.hist_numeric(cell_kpi, "mean_prb", "Cells: distribution of mean PRB (%)", bin_step=2.0),
            use_container_width=True,
        )

    b1, b2 = st.columns(2)
    with b1:
        st.altair_chart(
            plots.hist_numeric(cell_kpi, "mean_bout_minutes", "Cells: mean deep-sleep bout duration (minutes)", bin_step=15.0),
            use_container_width=True,
        )
    with b2:
        st.altair_chart(
            plots.hist_numeric(cell_kpi, "bouts_per_day", "Cells: deep-sleep bouts per day", bin_step=0.5),
            use_container_width=True,
        )

    st.altair_chart(
        plots.scatter_xy(
            cell_kpi,
            x="mean_prb",
            y="p_sleep",
            title="Cells: mean PRB vs P(sleep_on) (downsampled if large)",
            tooltip_cols=["Base Station ID", "Cell ID"],
        ),
        use_container_width=True,
    )

    st.markdown("### B) Per-base-station heterogeneity")
    d1, d2 = st.columns(2)
    with d1:
        st.altair_chart(
            plots.hist_numeric(bs_kpi, "eco_saved_kWh", "Base Stations: distribution of eco saved (kWh)", bin_step=None),
            use_container_width=True,
        )
    with d2:
        st.altair_chart(
            plots.hist_numeric(bs_kpi, "eco_saved_pct", "Base Stations: distribution of eco saved (%)", bin_step=1.0),
            use_container_width=True,
        )

    st.altair_chart(
        plots.scatter_xy(
            bs_kpi.rename(columns={bs_col: "Base Station ID"}),
            x="mean_prb",
            y="eco_saved_pct",
            title="Base Stations: mean PRB vs eco saved (%) (downsampled if large)",
            tooltip_cols=["Base Station ID"],
        ),
        use_container_width=True,
    )

    with st.expander("Show per-cell KPI table", expanded=False):
        st.dataframe(cell_kpi.sort_values("eco_saved_kWh", ascending=False).head(1000), use_container_width=True)

    with st.expander("Show per-base-station KPI table", expanded=False):
        st.dataframe(bs_kpi.sort_values("eco_saved_kWh", ascending=False), use_container_width=True)

    st.markdown("**Summary (JSON)**")
    st.json(
        {
            "dataset": dataset,
            "time_of_day": tod,
            "window": window_label,
            "threshold_scope": threshold_scope,
            "n_cells": int(len(cell_kpi)),
            "n_base_stations": int(len(bs_kpi)),
            "cell_metrics": {
                "cv_p_sleep": _cv(cell_kpi["p_sleep"]),
                "cv_mean_prb": _cv(cell_kpi["mean_prb"]),
            },
            "bs_metrics": {
                "cv_eco_saved_kwh": _cv(bs_kpi["eco_saved_kWh"]),
                "cv_eco_saved_pct": _cv(bs_kpi["eco_saved_pct"]),
            },
        }
    )
