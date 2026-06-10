"""Missing-value handling strategies.

Every function returns a *new* dataframe (never mutates the input) so the
session-state history can keep clean snapshots for undo.
"""
from __future__ import annotations

import pandas as pd


def drop_missing_rows(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Drop rows containing missing values (optionally only within ``subset``)."""
    return df.dropna(subset=subset).reset_index(drop=True)


def drop_missing_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns whose missing fraction exceeds ``threshold`` (0–1)."""
    keep = df.columns[df.isna().mean() <= threshold]
    return df[keep].copy()


def impute_mean(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill missing numeric values with each column's mean."""
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(out[col].mean())
    return out


def impute_median(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill missing numeric values with each column's median."""
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(out[col].median())
    return out


def impute_mode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill missing values with each column's most frequent value."""
    out = df.copy()
    for col in columns:
        mode = out[col].mode(dropna=True)
        if not mode.empty:
            out[col] = out[col].fillna(mode.iloc[0])
    return out


def impute_constant(df: pd.DataFrame, columns: list[str], value: object) -> pd.DataFrame:
    """Fill missing values in the chosen columns with a constant."""
    out = df.copy()
    for col in columns:
        out[col] = out[col].fillna(value)
    return out
