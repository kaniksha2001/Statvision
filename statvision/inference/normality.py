"""Normality testing battery.

Bundles five complementary tests so users can triangulate normality rather than
trust a single statistic. Returns a list of uniform result dicts.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from statvision.core.utils import decision_text


def shapiro_wilk(data: np.ndarray, alpha: float = 0.05) -> dict:
    """Shapiro-Wilk test (most powerful for small/medium samples)."""
    stat, p = stats.shapiro(data)
    return _result("Shapiro-Wilk", stat, p, alpha)


def ks_normal(data: np.ndarray, alpha: float = 0.05) -> dict:
    """Kolmogorov-Smirnov test against a normal fitted to the sample."""
    mu, sd = np.mean(data), np.std(data, ddof=1)
    stat, p = stats.kstest(data, "norm", args=(mu, sd))
    return _result("Kolmogorov-Smirnov", stat, p, alpha)


def anderson_normal(data: np.ndarray) -> dict:
    """Anderson-Darling test for normality (critical-value based)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        res = stats.anderson(data, dist="norm")
    levels = list(res.significance_level)
    idx = levels.index(5.0) if 5.0 in levels else len(levels) // 2
    crit = float(res.critical_values[idx])
    reject = res.statistic > crit
    return {
        "test": "Anderson-Darling",
        "statistic": float(res.statistic),
        "p_value": float("nan"),
        "critical_value_5pct": crit,
        "decision": "Reject H₀" if reject else "Fail to reject H₀",
        "interpretation": _interpret(reject),
    }


def jarque_bera(data: np.ndarray, alpha: float = 0.05) -> dict:
    """Jarque-Bera test based on sample skewness and kurtosis."""
    stat, p = stats.jarque_bera(data)
    return _result("Jarque-Bera", stat, p, alpha)


def dagostino_k2(data: np.ndarray, alpha: float = 0.05) -> dict:
    """D'Agostino-Pearson K² omnibus test."""
    stat, p = stats.normaltest(data)
    return _result("D'Agostino K²", float(stat), float(p), alpha)


def run_all(data: np.ndarray, alpha: float = 0.05) -> list[dict]:
    """Run every normality test and return the collected results."""
    tests = [shapiro_wilk, ks_normal, jarque_bera, dagostino_k2]
    results = [t(data, alpha) for t in tests]
    results.insert(2, anderson_normal(data))
    return results


# --- helpers ---------------------------------------------------------------
def _interpret(reject: bool) -> str:
    return ("Evidence against normality (data likely non-normal)."
            if reject else "No significant departure from normality detected.")


def _result(name: str, stat: float, p: float, alpha: float) -> dict:
    reject = (not np.isnan(p)) and p < alpha
    return {
        "test": name,
        "statistic": float(stat),
        "p_value": float(p),
        "decision": decision_text(p, alpha),
        "interpretation": _interpret(reject),
    }
