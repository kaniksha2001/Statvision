"""Numeric column transformations.

Scaling uses scikit-learn so behaviour matches the wider ML ecosystem, while
the log transform guards against non-positive inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def standardize(df: pd.DataFrame, columns: list[str], suffix: str = "_z") -> pd.DataFrame:
    """Z-score standardisation: (x - mean) / std, via ``StandardScaler``."""
    out = df.copy()
    scaler = StandardScaler()
    out[[c + suffix for c in columns]] = scaler.fit_transform(out[columns])
    return out


def normalize(df: pd.DataFrame, columns: list[str], suffix: str = "_norm") -> pd.DataFrame:
    """L2 unit-norm normalisation applied column-wise."""
    out = df.copy()
    for col in columns:
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        norm = np.linalg.norm(values[~np.isnan(values)])
        out[col + suffix] = values / norm if norm else values
    return out


def min_max_scale(df: pd.DataFrame, columns: list[str], suffix: str = "_mm") -> pd.DataFrame:
    """Min-max scaling to the [0, 1] range via ``MinMaxScaler``."""
    out = df.copy()
    scaler = MinMaxScaler()
    out[[c + suffix for c in columns]] = scaler.fit_transform(out[columns])
    return out


def log_transform(df: pd.DataFrame, columns: list[str], suffix: str = "_log") -> pd.DataFrame:
    """Apply log1p (``log(1 + x)``) after shifting so the minimum is >= 0.

    Shifting keeps the transform defined for zero / negative values while still
    being monotonic, which is the behaviour most analysts expect.
    """
    out = df.copy()
    for col in columns:
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        shift = -finite.min() if finite.size and finite.min() < 0 else 0.0
        out[col + suffix] = np.log1p(values + shift)
    return out
