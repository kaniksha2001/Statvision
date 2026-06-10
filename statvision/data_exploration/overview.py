"""Dataset overview: shape, dtypes, memory and per-column metadata."""
from __future__ import annotations

import numpy as np
import pandas as pd

from statvision.core.utils import memory_usage_human


def overview(df: pd.DataFrame) -> dict[str, object]:
    """Return high-level dataset metrics for the dashboard cards."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory": memory_usage_human(df),
        "numeric": int(df.select_dtypes(include=[np.number]).shape[1]),
        "categorical": int(df.select_dtypes(include=["object", "category", "bool"]).shape[1]),
        "total_cells": int(df.size),
        "total_missing": int(df.isna().sum().sum()),
    }


def column_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Build a tidy per-column metadata table.

    Columns: name, dtype, non-null count, missing count, missing %, unique
    values and a small sample of example values.
    """
    rows: list[dict[str, object]] = []
    n = len(df)
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        examples = series.dropna().unique()[:3]
        rows.append(
            {
                "Column": str(col),
                "Dtype": str(series.dtype),
                "Non-Null": int(n - missing),
                "Missing": missing,
                "Missing %": round(100 * missing / n, 2) if n else 0.0,
                "Unique": int(series.nunique(dropna=True)),
                "Examples": ", ".join(map(lambda x: str(x), examples)),
            }
        )
    return pd.DataFrame(rows)


def filter_dataframe(
    df: pd.DataFrame,
    search: str = "",
    column: str | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """Apply a free-text search, optional column scoping and sorting.

    The search is a case-insensitive substring match across either a single
    chosen column or every column when ``column`` is ``None``/``"All columns"``.
    """
    result = df
    if search:
        if column and column != "All columns":
            mask = result[column].astype(str).str.contains(search, case=False, na=False)
        else:
            mask = result.apply(
                lambda r: r.astype(str).str.contains(search, case=False, na=False).any(),
                axis=1,
            )
        result = result[mask]
    if sort_by and sort_by in result.columns:
        result = result.sort_values(by=sort_by, ascending=ascending, kind="mergesort")
    return result
