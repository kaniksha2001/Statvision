"""Shared, dependency-light helper utilities used across every module.

These functions intentionally have no Streamlit dependency so they can be unit
tested in isolation and reused by the reporting engine.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return the names of numeric (int/float) columns."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return the names of object / categorical / boolean columns."""
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def datetime_columns(df: pd.DataFrame) -> list[str]:
    """Return the names of datetime columns."""
    return df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()


def clean_series(series: pd.Series) -> np.ndarray:
    """Return a 1-D float array of a numeric series with NaNs removed."""
    return pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)


def fmt(value: float, digits: int = 4) -> str:
    """Format a float for display, gracefully handling NaN / inf / None."""
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(f):
        return "NaN"
    if np.isinf(f):
        return "∞" if f > 0 else "-∞"
    if f == 0:
        return "0"
    if abs(f) < 1e-4 or abs(f) >= 1e6:
        return f"{f:.{digits}e}"
    return f"{f:,.{digits}f}".rstrip("0").rstrip(".")


def memory_usage_human(df: pd.DataFrame) -> str:
    """Return a human readable memory footprint for a dataframe."""
    nbytes = int(df.memory_usage(deep=True).sum())
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} TB"


def decision_text(p_value: float, alpha: float) -> str:
    """Return a standard reject / fail-to-reject decision string."""
    if p_value is None or np.isnan(p_value):
        return "Indeterminate (p-value unavailable)"
    return ("Reject H₀" if p_value < alpha
            else "Fail to reject H₀")


def safe_len(x: Iterable) -> int:
    """Length helper that tolerates generators / None."""
    try:
        return len(x)  # type: ignore[arg-type]
    except TypeError:
        return sum(1 for _ in x)
