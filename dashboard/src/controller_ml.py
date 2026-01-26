# src/controller_ml.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from src.config import DT_HOURS, resolve_col
from src.ml.features_shared import FeatureSpec, make_X


@dataclass(frozen=True)
class MLControllerSpec:
    tau_on: float = 0.80
    tau_off: float = 0.70
    hysteresis_enabled: bool = True

    def resolved_tau_off(self) -> float:
        if not self.hysteresis_enabled:
            return float(self.tau_on)
        return float(min(self.tau_on, self.tau_off))


def load_model(model_path: str | Path):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"ML model not found: {p}")
    return joblib.load(str(p))


def predict_p_sleep_on(df_prepared: pd.DataFrame, model, feature_spec: FeatureSpec) -> np.ndarray:
    X = make_X(df_prepared, spec=feature_spec, return_feature_names=False)
    p = model.predict_proba(X)[:, 1].astype(float)
    return np.clip(p, 0.0, 1.0)


def decide_state_prob_hysteresis(p: float, prev: str, tau_on: float, tau_off: float) -> str:
    ps = prev if prev in {"FULL", "ECO"} else "FULL"
    if np.isnan(p):
        return ps
    if ps == "FULL":
        return "ECO" if p >= tau_on else "FULL"
    return "FULL" if p <= tau_off else "ECO"


def apply_simulation_only(
    df_with_probs: pd.DataFrame,
    controller: MLControllerSpec,
    alpha: float,
) -> pd.DataFrame:
    """
    OPTIMIZED: Replaces .apply() with a faster transform logic.
    """
    df_out = df_with_probs.copy()
    
    bs_col = resolve_col(df_out, "bs")
    cell_col = resolve_col(df_out, "cell")
    rru_col = resolve_col(df_out, "rru_w")

    tau_on = float(controller.tau_on)
    tau_off = float(controller.resolved_tau_off())

    # Sort once for time-continuity
    df_out = df_out.sort_values([bs_col, cell_col, "DayIndex", "tod_bin"])

    # Fast logic operating on numpy arrays
    def fast_hysteresis_vec(p_vals, t_on, t_off):
        n = len(p_vals)
        states = np.empty(n, dtype=object)
        prev = "FULL"
        for i in range(n):
            p = p_vals[i]
            if np.isnan(p):
                states[i] = prev
                continue
            
            if prev == "FULL":
                curr = "ECO" if p >= t_on else "FULL"
            else: # prev == "ECO"
                curr = "FULL" if p <= t_off else "ECO"
            
            states[i] = curr
            prev = curr
        return states

    # Apply the logic efficiently
    df_out["State"] = (
        df_out.groupby([bs_col, cell_col, "DayIndex"], sort=False)["p_sleep_on"]
        .transform(lambda x: fast_hysteresis_vec(x.to_numpy(), tau_on, tau_off))
    )

    # Vectorized energy math
    rru_vec = pd.to_numeric(df_out[rru_col], errors="coerce").fillna(0.0).to_numpy()
    df_out["baseline_Wh"] = rru_vec * DT_HOURS
    is_eco = (df_out["State"].to_numpy() == "ECO").astype(float)
    df_out["eco_saved_Wh"] = (float(alpha) * df_out["baseline_Wh"].to_numpy()) * is_eco
    
    # UI Schema compatibility
    if "p30" not in df_out.columns: df_out["p30"] = np.nan
    if "p70" not in df_out.columns: df_out["p70"] = np.nan
    
    return df_out


def apply_ml_controller_and_simulation(
    df: pd.DataFrame,
    *,
    model,
    feature_spec: FeatureSpec,
    controller: MLControllerSpec,
    alpha: float,
) -> pd.DataFrame:
    """
    Full path: Prediction -> Hysteresis -> Energy -> Schema Alignment.
    """
    df_out = df.copy()
    
    # 1. Prediction
    p = predict_p_sleep_on(df_out, model=model, feature_spec=feature_spec)
    df_out["p_sleep_on"] = p

    # 2. Simulation (Hysteresis + Energy + Placeholders)
    return apply_simulation_only(df_out, controller=controller, alpha=alpha)