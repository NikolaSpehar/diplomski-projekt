from __future__ import annotations

from typing import Iterable, Optional
import pandas as pd

DT_SECONDS = 1800.0
DT_HOURS = DT_SECONDS / 3600.0  # 0.5 hours for 30-min bins

# Canonical column contract:
COL = {
    "bs": "Base Station ID",
    "cell": "Cell ID",
    "ts": "Timestamp",
    "prb": "PRB Usage Ratio (%)",
    "traffic_kb": "Traffic Volume (KByte)",
    "users": "Number of Users",
    "bbu_w": "BBU Energy (W)",
    "rru_w": "RRU Energy (W)",
    "ds_ms": "Deep Sleep Time (Millisecond)",
}


def _norm_col(s: str) -> str:
    s = str(s).replace("\ufeff", "").replace("\xa0", " ").strip()
    s = " ".join(s.split())
    return s


def resolve_col(df: pd.DataFrame, key: str) -> str:
    if key not in COL:
        raise KeyError(f"Unknown COL key: {key!r}. Known keys: {sorted(COL.keys())}")

    target = COL[key]
    if target in df.columns:
        return target

    target_n = _norm_col(target)
    candidates = {c: _norm_col(c) for c in df.columns}
    for c, cn in candidates.items():
        if cn == target_n:
            return c

    raise KeyError(
        f"Column mapping mismatch for key='{key}'. "
        f"Expected {target!r} not found in df.columns. "
        f"Available columns include: {list(df.columns)[:20]} ..."
    )


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    normalized = [_norm_col(c) for c in out.columns]
    out.columns = normalized

    rename_map: dict[str, str] = {}
    col_set = set(out.columns)

    for _, canonical in COL.items():
        canon_norm = _norm_col(canonical)
        if canon_norm in col_set and canon_norm != canonical:
            rename_map[canon_norm] = canonical
        elif canon_norm in col_set and canon_norm == canonical:
            continue
        else:
            for existing in out.columns:
                if _norm_col(existing) == canon_norm:
                    rename_map[existing] = canonical
                    break

    if rename_map:
        out = out.rename(columns=rename_map)

    return out


def validate_schema(
    df: pd.DataFrame,
    required_keys: Optional[Iterable[str]] = None,
    required_cols: Optional[Iterable[str]] = None,
    where: str = "validate_schema",
    max_preview_cols: int = 20,
) -> dict:
    required_keys = list(required_keys or [])
    required_cols = list(required_cols or [])

    missing: list[str] = []
    resolved: dict[str, str] = {}

    for k in required_keys:
        try:
            resolved_name = resolve_col(df, k)
            resolved[k] = resolved_name
        except Exception:
            resolved[k] = "<unresolved>"
            missing.append(f"COL['{k}']={COL.get(k)!r}")

    for c in required_cols:
        if c not in df.columns:
            missing.append(c)

    if missing:
        preview_cols = list(df.columns)[:max_preview_cols]
        raise ValueError(
            f"{where}: Missing required columns: {missing}. "
            f"Available columns (first {max_preview_cols}): {preview_cols}"
        )

    info = {
        "where": where,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "resolved_keys": resolved,
    }
    return info
