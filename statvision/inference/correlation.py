"""Correlation analysis: Pearson, Spearman and Kendall coefficients."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from statvision.core.utils import numeric_columns

_METHODS = {
    "pearson": ("Pearson", stats.pearsonr, "linear relationship"),
    "spearman": ("Spearman", stats.spearmanr, "monotonic (rank) relationship"),
    "kendall": ("Kendall", stats.kendalltau, "ordinal concordance"),
}


def pairwise_correlation(x: np.ndarray, y: np.ndarray, method: str = "pearson",
                         alpha: float = 0.05) -> dict:
    """Compute a single correlation coefficient with its significance test."""
    name, func, desc = _METHODS[method]
    coef, p = func(x, y)
    strength = _strength(abs(coef))
    return {
        "method": name,
        "coefficient": float(coef),
        "p_value": float(p),
        "n": int(min(len(x), len(y))),
        "significant": bool(p < alpha),
        "interpretation": (
            f"{strength} {desc} (r = {coef:.3f}); "
            f"{'statistically significant' if p < alpha else 'not significant'} at α={alpha}."
        ),
    }


def correlation_matrix(df: pd.DataFrame, method: str = "pearson",
                       columns: list[str] | None = None) -> pd.DataFrame:
    """Return a correlation matrix for numeric columns using the chosen method."""
    columns = columns or numeric_columns(df)
    return df[columns].corr(method=method)


def _strength(r: float) -> str:
    """Map an absolute coefficient to a qualitative strength label."""
    if r < 0.1:
        return "Negligible"
    if r < 0.3:
        return "Weak"
    if r < 0.5:
        return "Moderate"
    if r < 0.7:
        return "Strong"
    return "Very strong"
