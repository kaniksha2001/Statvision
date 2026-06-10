"""Descriptive statistics.

All measures are computed with NumPy / SciPy. Sample (ddof=1) conventions are
used for variance and standard deviation, matching SPSS/Minitab defaults.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from statvision.core.utils import clean_series, numeric_columns


# --- Central tendency -------------------------------------------------------
def mean(data: np.ndarray) -> float:
    return float(np.mean(data))


def median(data: np.ndarray) -> float:
    return float(np.median(data))


def mode(data: np.ndarray) -> float:
    """Most frequent value (first when multimodal)."""
    result = stats.mode(data, keepdims=False)
    return float(np.atleast_1d(result.mode)[0])


def trimmed_mean(data: np.ndarray, proportion: float = 0.1) -> float:
    """Mean after trimming ``proportion`` from each tail."""
    return float(stats.trim_mean(data, proportiontocut=proportion))


# --- Dispersion -------------------------------------------------------------
def value_range(data: np.ndarray) -> float:
    return float(np.max(data) - np.min(data))


def variance(data: np.ndarray) -> float:
    return float(np.var(data, ddof=1))


def std_dev(data: np.ndarray) -> float:
    return float(np.std(data, ddof=1))


def iqr(data: np.ndarray) -> float:
    return float(stats.iqr(data))


def coefficient_of_variation(data: np.ndarray) -> float:
    """CV = std / mean (returns NaN when the mean is zero)."""
    m = np.mean(data)
    return float(np.std(data, ddof=1) / m) if m != 0 else float("nan")


# --- Position ---------------------------------------------------------------
def quartiles(data: np.ndarray) -> dict[str, float]:
    q1, q2, q3 = np.percentile(data, [25, 50, 75])
    return {"Q1": float(q1), "Q2": float(q2), "Q3": float(q3)}


def percentiles(data: np.ndarray, ps: list[int] | None = None) -> dict[str, float]:
    ps = ps or [5, 10, 25, 50, 75, 90, 95]
    values = np.percentile(data, ps)
    return {f"P{p}": float(v) for p, v in zip(ps, values)}


def deciles(data: np.ndarray) -> dict[str, float]:
    points = list(range(10, 100, 10))
    values = np.percentile(data, points)
    return {f"D{p // 10}": float(v) for p, v in zip(points, values)}


# --- Shape ------------------------------------------------------------------
def skewness(data: np.ndarray) -> float:
    """Sample skewness (bias-corrected)."""
    return float(stats.skew(data, bias=False))


def kurtosis(data: np.ndarray) -> float:
    """Excess kurtosis (normal distribution = 0)."""
    return float(stats.kurtosis(data, fisher=True, bias=False))


# --- Summary table ----------------------------------------------------------
def summary_table(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Build a tidy summary statistics table for the chosen numeric columns."""
    columns = columns or numeric_columns(df)
    rows: list[dict[str, object]] = []
    for col in columns:
        data = clean_series(df[col])
        if data.size == 0:
            continue
        rows.append(
            {
                "Variable": col,
                "Count": int(data.size),
                "Mean": mean(data),
                "Median": median(data),
                "Min": float(np.min(data)),
                "Max": float(np.max(data)),
                "Std Dev": std_dev(data),
                "Variance": variance(data),
                "IQR": iqr(data),
                "Skewness": skewness(data),
                "Kurtosis": kurtosis(data),
            }
        )
    return pd.DataFrame(rows)


def full_describe(data: np.ndarray) -> dict[str, float]:
    """Return every measure for a single series as a flat dictionary."""
    out: dict[str, float] = {
        "Mean": mean(data),
        "Median": median(data),
        "Mode": mode(data),
        "Trimmed Mean (10%)": trimmed_mean(data),
        "Range": value_range(data),
        "Variance": variance(data),
        "Std Dev": std_dev(data),
        "IQR": iqr(data),
        "Coef. of Variation": coefficient_of_variation(data),
        "Skewness": skewness(data),
        "Kurtosis": kurtosis(data),
    }
    out.update(quartiles(data))
    out.update(percentiles(data))
    return out
