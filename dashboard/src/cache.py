# src/cache.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import streamlit as st

from src.io import load_csv
from src.config import canonicalize_columns, validate_schema
from src.features import prepare_5g
from src.aggregations import (
    apply_policy_and_simulation,
    slice_time_of_day,
    bs_summary,
    compute_threshold_table_for_scope,
)
from src.kpis import per_cell_kpis


def _cache_dir_for_source(source_path: str) -> Path:
    p = Path(source_path)
    # Keep cache adjacent to the source data file
    d = p.parent / "_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parquet_path_for_csv(csv_path: str) -> Path:
    p = Path(csv_path)
    cache_dir = _cache_dir_for_source(csv_path)
    return cache_dir / f"{p.stem}.parquet"


def _thr_parquet_path(csv_or_parquet_path: str, threshold_scope: str) -> Path:
    p = Path(csv_or_parquet_path)
    cache_dir = _cache_dir_for_source(str(p))
    # Use dataset stem so CSV and its cached parquet share the same threshold cache
    stem = p.stem.replace(".parquet", "")
    safe_scope = threshold_scope.replace("/", "_")
    return cache_dir / f"{stem}.thresholds.{safe_scope}.parquet"


def _read_parquet(path: Path) -> pd.DataFrame:
    # Pandas requires pyarrow or fastparquet for parquet IO
    return pd.read_parquet(path)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


def ensure_parquet_dataset(csv_path: str) -> str:
    """
    Ensure a Parquet version of the CSV exists.
    Returns the Parquet path as string (preferred read path).
    """
    pq = _parquet_path_for_csv(csv_path)
    if pq.exists():
        return str(pq)

    raw = load_csv(csv_path)
    raw = canonicalize_columns(raw)
    validate_schema(raw, required_keys=["ts", "bs", "cell", "prb", "traffic_kb", "rru_w"], where="ensure_parquet_dataset:raw")

    _write_parquet(raw, pq)
    return str(pq)


@st.cache_data(show_spinner=False)
def get_raw_df(source_path: str) -> pd.DataFrame:
    """
    Fast path:
      - If source is CSV: convert once to Parquet in _cache/ and load Parquet thereafter.
      - If source is already Parquet: read directly.
    """
    p = Path(source_path)
    if p.suffix.lower() == ".csv":
        pq_path = ensure_parquet_dataset(source_path)
        df = _read_parquet(Path(pq_path))
    elif p.suffix.lower() == ".parquet":
        df = _read_parquet(p)
    else:
        # fall back to CSV reader for unknown extensions
        df = load_csv(source_path)
        df = canonicalize_columns(df)

    # safety: schema check (lightweight)
    validate_schema(df, required_keys=["ts", "bs", "cell", "prb", "traffic_kb", "rru_w"], where="cache:get_raw_df")
    return df


@st.cache_data(show_spinner=False)
def get_prepared_df(source_path: str) -> pd.DataFrame:
    raw = get_raw_df(source_path)
    df = prepare_5g(raw)
    validate_schema(
        df,
        required_keys=["ts", "bs", "cell", "prb", "traffic_kb", "rru_w"],
        required_cols=["tod_bin", "DayIndex", "sleep_on", "sleep_frac"],
        where="cache:get_prepared_df",
    )
    return df


def ensure_thresholds_parquet(source_path: str, threshold_scope: str) -> str:
    """
    Compute thresholds once per dataset+scope and persist in _cache/ as Parquet.
    Returns path to thresholds parquet.
    """
    thr_path = _thr_parquet_path(source_path, threshold_scope)
    if thr_path.exists():
        return str(thr_path)

    df = get_prepared_df(source_path)

    thr = compute_threshold_table_for_scope(df, threshold_scope=threshold_scope)
    # Persist
    _write_parquet(thr, thr_path)
    return str(thr_path)


@st.cache_data(show_spinner=False)
def get_thresholds_df(source_path: str, threshold_scope: str) -> pd.DataFrame:
    thr_pq = ensure_thresholds_parquet(source_path, threshold_scope)
    thr = _read_parquet(Path(thr_pq))
    return thr


@st.cache_data(show_spinner=False)
def get_policy_df(
    source_path: str,
    threshold_scope: str,
    alpha: float,
    hysteresis_enabled: bool,
    h_sleep: float,
    h_eco: float,
) -> pd.DataFrame:
    df = get_prepared_df(source_path)

    # Load persisted thresholds for this dataset+scope
    thr = get_thresholds_df(source_path, threshold_scope)

    out = apply_policy_and_simulation(
        df,
        threshold_scope=threshold_scope,
        alpha=alpha,
        hysteresis_enabled=hysteresis_enabled,
        h_sleep=h_sleep,
        h_eco=h_eco,
        thresholds_table=thr,  # key change: reuse precomputed thresholds
    )

    validate_schema(
        out,
        required_keys=["bs", "cell", "prb", "rru_w"],
        required_cols=["tod_bin", "DayIndex", "p30", "p70", "State", "baseline_Wh", "eco_saved_Wh"],
        where="cache:get_policy_df",
    )
    return out


@st.cache_data(show_spinner=False)
def get_view_df(
    source_path: str,
    threshold_scope: str,
    alpha: float,
    hysteresis_enabled: bool,
    h_sleep: float,
    h_eco: float,
    window_mode: str,
    tod_bin: int,
) -> pd.DataFrame:
    df_policy = get_policy_df(source_path, threshold_scope, alpha, hysteresis_enabled, h_sleep, h_eco)
    if window_mode == "Selected time-of-day":
        return slice_time_of_day(df_policy, tod_bin=tod_bin)
    return df_policy


@st.cache_data(show_spinner=False)
def get_bs_summary_cached(
    source_path: str,
    threshold_scope: str,
    alpha: float,
    hysteresis_enabled: bool,
    h_sleep: float,
    h_eco: float,
    window_mode: str,
    tod_bin: int,
) -> pd.DataFrame:
    df_view = get_view_df(source_path, threshold_scope, alpha, hysteresis_enabled, h_sleep, h_eco, window_mode, tod_bin)
    return bs_summary(df_view)


@st.cache_data(show_spinner=False)
def get_per_cell_kpis_cached(
    source_path: str,
    threshold_scope: str,
    alpha: float,
    hysteresis_enabled: bool,
    h_sleep: float,
    h_eco: float,
    window_mode: str,
    tod_bin: int,
) -> pd.DataFrame:
    df_view = get_view_df(source_path, threshold_scope, alpha, hysteresis_enabled, h_sleep, h_eco, window_mode, tod_bin)
    return per_cell_kpis(df_view)
