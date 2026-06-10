"""Data quality assessment and automated profiling.

Provides missing-value accounting, duplicate detection, IQR/z-score outlier
summaries and a consolidated profiling report that the Exploration page renders
and the report engine can embed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from statvision.core.utils import clean_series, numeric_columns


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and percentages, sorted worst-first."""
    n = len(df)
    counts = df.isna().sum()
    out = pd.DataFrame(
        {
            "Column": counts.index.astype(str),
            "Missing": counts.values.astype(int),
            "Missing %": (100 * counts.values / n).round(2) if n else 0.0,
        }
    )
    return out.sort_values("Missing", ascending=False).reset_index(drop=True)


def duplicate_report(df: pd.DataFrame) -> dict[str, float]:
    """Return duplicate row count and percentage."""
    n = len(df)
    dup = int(df.duplicated().sum())
    return {"duplicates": dup, "duplicate_pct": round(100 * dup / n, 2) if n else 0.0}


def outlier_summary(df: pd.DataFrame, method: str = "iqr", z_thresh: float = 3.0,
                    iqr_factor: float = 1.5) -> pd.DataFrame:
    """Summarise outliers for each numeric column.

    ``method`` is ``"iqr"`` (Tukey fences) or ``"zscore"``. Returns one row per
    numeric column with the count and percentage of flagged observations plus
    the lower/upper bound used.
    """
    rows: list[dict[str, object]] = []
    for col in numeric_columns(df):
        data = clean_series(df[col])
        if data.size == 0:
            continue
        if method == "zscore":
            mu, sd = data.mean(), data.std(ddof=1)
            if sd == 0 or np.isnan(sd):
                lower, upper = mu, mu
                count = 0
            else:
                z = np.abs((data - mu) / sd)
                count = int((z > z_thresh).sum())
                lower, upper = mu - z_thresh * sd, mu + z_thresh * sd
        else:  # IQR / Tukey
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
            count = int(((data < lower) | (data > upper)).sum())
        rows.append(
            {
                "Column": str(col),
                "Outliers": count,
                "Outlier %": round(100 * count / data.size, 2),
                "Lower Bound": round(float(lower), 4),
                "Upper Bound": round(float(upper), 4),
            }
        )
    return pd.DataFrame(rows)


def outlier_mask(series: pd.Series, method: str = "iqr", z_thresh: float = 3.0,
                 iqr_factor: float = 1.5) -> pd.Series:
    """Return a boolean mask flagging outliers in a single numeric series."""
    numeric = pd.to_numeric(series, errors="coerce")
    if method == "zscore":
        mu, sd = numeric.mean(), numeric.std(ddof=1)
        if sd == 0 or np.isnan(sd):
            return pd.Series(False, index=series.index)
        return (numeric - mu).abs() / sd > z_thresh
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    return (numeric < q1 - iqr_factor * iqr) | (numeric > q3 + iqr_factor * iqr)


def profiling_report(df: pd.DataFrame) -> dict[str, object]:
    """Assemble a complete automated profile of the dataset.

    Returns a dictionary with overview metrics, the missing/duplicate/outlier
    tables and a per-column descriptive snippet, ready for display and export.
    """
    from statvision.data_exploration.overview import overview, column_metadata

    return {
        "overview": overview(df),
        "columns": column_metadata(df),
        "missing": missing_report(df),
        "duplicates": duplicate_report(df),
        "outliers": outlier_summary(df),
        "describe": df.describe(include="all").transpose(),
    }
