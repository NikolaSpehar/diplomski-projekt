# src/kpis.py
from __future__ import annotations

import pandas as pd
import numpy as np

from .config import resolve_col, DT_SECONDS


def bout_ids(is_sleep: pd.Series) -> pd.Series:
    s = is_sleep.astype(int)
    start = (s == 1) & (s.shift(1, fill_value=0) == 0)
    bout = start.cumsum()
    return bout.where(s == 1, 0)


def per_cell_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-cell KPIs meant for heterogeneity analysis.

    Requires:
      - sleep_on, sleep_frac, tod_bin, DayIndex
      - baseline_Wh, eco_saved_Wh (policy output)
    """
    bs_col = resolve_col(df, "bs")
    cell_col = resolve_col(df, "cell")
    prb_col = resolve_col(df, "prb")
    rru_col = resolve_col(df, "rru_w")

    keys = [bs_col, cell_col]
    g = df.sort_values(keys + ["DayIndex", "tod_bin"]).copy()
    g["bout_id"] = g.groupby(keys)["sleep_on"].transform(bout_ids)

    def agg(x: pd.DataFrame) -> pd.Series:
        n = len(x)
        bout_sizes = x.loc[x["bout_id"] > 0, "bout_id"].value_counts()
        mean_bout_intervals = float(bout_sizes.mean()) if len(bout_sizes) else 0.0
        mean_bout_minutes = mean_bout_intervals * (DT_SECONDS / 60.0)

        n_days = int(x["DayIndex"].nunique()) if "DayIndex" in x.columns else 1
        bouts_per_day = float(len(bout_sizes) / max(1, n_days))

        baseline = float(np.nansum(x["baseline_Wh"].values)) if "baseline_Wh" in x.columns else 0.0
        saved = float(np.nansum(x["eco_saved_Wh"].values)) if "eco_saved_Wh" in x.columns else 0.0

        return pd.Series({
            "n_rows": int(n),
            "n_days": int(n_days),
            "p_sleep": float(x["sleep_on"].mean()) if n else 0.0,
            "f_sleep": float(x["sleep_frac"].mean()) if n else 0.0,
            "n_bouts": int(len(bout_sizes)),
            "bouts_per_day": float(bouts_per_day),
            "mean_bout_intervals": float(mean_bout_intervals),
            "mean_bout_minutes": float(mean_bout_minutes),
            "mean_prb": float(pd.to_numeric(x[prb_col], errors="coerce").mean()),
            "mean_rru_w": float(pd.to_numeric(x[rru_col], errors="coerce").mean()),
            "baseline_Wh": float(baseline),
            "eco_saved_Wh": float(saved),
            "eco_saved_pct_of_baseline": (100.0 * saved / baseline) if baseline > 0 else 0.0,
        })

    out = g.groupby(keys, dropna=False).apply(agg, include_groups=False).reset_index()

    out = out.rename(columns={bs_col: "Base Station ID", cell_col: "Cell ID"})
    return out


def bs_level_gt_vs_pred(df_bs: pd.DataFrame, state_col: str = "State") -> dict:
    bs_col = resolve_col(df_bs, "bs")

    by_bs = df_bs.groupby(bs_col, dropna=False).agg(
        gt_sleep_any=("sleep_on", lambda s: bool((s == True).any())),
        pred_sleep=(state_col, lambda s: bool((s == "SLEEP").any())),
    ).reset_index()

    tp = int(((by_bs["pred_sleep"]) & (by_bs["gt_sleep_any"])).sum())
    fp = int(((by_bs["pred_sleep"]) & (~by_bs["gt_sleep_any"])).sum())
    fn = int(((~by_bs["pred_sleep"]) & (by_bs["gt_sleep_any"])).sum())
    tn = int(((~by_bs["pred_sleep"]) & (~by_bs["gt_sleep_any"])).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def calculate_risk_metrics(df: pd.DataFrame, prb_threshold: float = 20.0) -> dict:
    """
    Risk Proxy:
      An interval is 'High Risk' if the Controller is in ECO mode
      BUT PRB load is > prb_threshold.

    IMPORTANT: optimized to avoid big DataFrame copies on every rerun.
    """
    if df is None or df.empty:
        return {
            "risk_intervals": 0,
            "risk_percent_total": 0.0,
            "risk_percent_eco": 0.0,
            "eco_intervals": 0,
        }

    if "State" not in df.columns:
        return {
            "risk_intervals": 0,
            "risk_percent_total": 0.0,
            "risk_percent_eco": 0.0,
            "eco_intervals": 0,
        }

    prb_col = resolve_col(df, "prb")

    # Extract minimal columns as arrays (no .copy(), no new dataframe)
    state = df["State"].astype(str).to_numpy()
    prb = pd.to_numeric(df[prb_col], errors="coerce").to_numpy()

    valid = ~np.isnan(prb) & (state != "UNKNOWN") & (state != "nan")
    if not np.any(valid):
        return {
            "risk_intervals": 0,
            "risk_percent_total": 0.0,
            "risk_percent_eco": 0.0,
            "eco_intervals": 0,
        }

    state_v = state[valid]
    prb_v = prb[valid]

    is_eco = (state_v == "ECO")
    n_total = int(len(state_v))
    n_eco = int(np.sum(is_eco))

    if n_total == 0:
        return {
            "risk_intervals": 0,
            "risk_percent_total": 0.0,
            "risk_percent_eco": 0.0,
            "eco_intervals": 0,
        }

    is_high_load = prb_v > float(prb_threshold)
    risk_mask = is_eco & is_high_load
    n_risk = int(np.sum(risk_mask))

    return {
        "risk_intervals": n_risk,
        "risk_percent_total": 100.0 * float(n_risk) / float(n_total),
        "risk_percent_eco": (100.0 * float(n_risk) / float(n_eco)) if n_eco > 0 else 0.0,
        "eco_intervals": n_eco,
    }
