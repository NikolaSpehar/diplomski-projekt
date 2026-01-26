# src/optimization.py
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st

from src.controller_ml import MLControllerSpec, apply_simulation_only
from src.kpis import calculate_risk_metrics
from src.config import resolve_col

def run_pareto_sweep(
    df_with_probs: pd.DataFrame,
    alpha: float,
    tau_on_range: list[float],
    tau_off_offset: float = 0.10,
    prb_threshold: float = 20.0,
    sample_frac: float = 0.20 # OPTIMIZATION: Use 20% of cells for the sweep
) -> pd.DataFrame:
    
    bs_col = resolve_col(df_with_probs, "bs")
    cell_col = resolve_col(df_with_probs, "cell")
    
    # Unique Cell Keys
    unique_cells = df_with_probs[[bs_col, cell_col]].drop_duplicates()
    n_sample = max(1, int(len(unique_cells) * sample_frac))
    sampled_keys = unique_cells.sample(n=n_sample, random_state=42)
    
    # Filter only the sample
    sweep_df = df_with_probs.merge(sampled_keys, on=[bs_col, cell_col])
    
    results = []
    prog = st.progress(0.0)
    
    for i, t_on in enumerate(tau_on_range):
        t_off = max(0.0, t_on - tau_off_offset)
        ctrl = MLControllerSpec(tau_on=t_on, tau_off=t_off, hysteresis_enabled=True)
        
        # Run the fast simulation
        sim_df = apply_simulation_only(sweep_df, controller=ctrl, alpha=alpha)
        
        # Metrics
        total_kwh = sim_df["eco_saved_Wh"].sum() / 1000.0
        risk_meta = calculate_risk_metrics(sim_df, prb_threshold=prb_threshold)
        
        results.append({
            "tau_on": t_on,
            "saved_kwh": total_kwh,
            "risk_pct": risk_meta["risk_percent_total"],
            "eco_coverage_pct": (100.0 * risk_meta["eco_intervals"] / len(sim_df))
        })
        prog.progress((i + 1) / len(tau_on_range))
    
    prog.empty()
    return pd.DataFrame(results)