"""Duplicate row detection and removal."""
from __future__ import annotations

import pandas as pd


def detect_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Return the subset of rows that are duplicates (keeping all occurrences)."""
    mask = df.duplicated(subset=subset, keep=False)
    return df[mask]


def remove_duplicates(df: pd.DataFrame, subset: list[str] | None = None,
                      keep: str = "first") -> pd.DataFrame:
    """Drop duplicate rows, retaining the ``keep`` occurrence (first/last)."""
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
