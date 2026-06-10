"""Confidence intervals for the mean, a proportion and the variance."""
from __future__ import annotations

import numpy as np
from scipy import stats


def mean_ci(data: np.ndarray, confidence: float = 0.95) -> dict:
    """Confidence interval for the population mean using Student's t.

    The t-distribution is used because the population variance is unknown, which
    is the realistic case for sample data.
    """
    n = data.size
    mean = float(np.mean(data))
    se = float(stats.sem(data))
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_crit * se
    return {
        "parameter": "Mean",
        "estimate": mean,
        "lower": mean - margin,
        "upper": mean + margin,
        "margin_of_error": margin,
        "confidence": confidence,
        "n": n,
        "std_error": se,
    }


def proportion_ci(successes: int, n: int, confidence: float = 0.95,
                  method: str = "wilson") -> dict:
    """Confidence interval for a proportion.

    Defaults to the Wilson score interval (better coverage than the normal
    approximation, especially for small samples or extreme proportions).
    """
    p_hat = successes / n
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)
    if method == "wald":
        se = np.sqrt(p_hat * (1 - p_hat) / n)
        margin = z * se
        lower, upper = p_hat - margin, p_hat + margin
    else:  # Wilson score
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        half = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
        lower, upper = center - half, center + half
        margin = half
    return {
        "parameter": "Proportion",
        "estimate": p_hat,
        "lower": max(0.0, float(lower)),
        "upper": min(1.0, float(upper)),
        "margin_of_error": float(margin),
        "confidence": confidence,
        "n": n,
        "method": method,
    }


def variance_ci(data: np.ndarray, confidence: float = 0.95) -> dict:
    """Confidence interval for the population variance using the chi-square dist."""
    n = data.size
    s2 = float(np.var(data, ddof=1))
    alpha = 1 - confidence
    chi2_lower = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
    chi2_upper = stats.chi2.ppf(alpha / 2, df=n - 1)
    lower = (n - 1) * s2 / chi2_lower
    upper = (n - 1) * s2 / chi2_upper
    return {
        "parameter": "Variance",
        "estimate": s2,
        "lower": float(lower),
        "upper": float(upper),
        "margin_of_error": float((upper - lower) / 2),
        "confidence": confidence,
        "n": n,
        "std_dev_lower": float(np.sqrt(lower)),
        "std_dev_upper": float(np.sqrt(upper)),
    }
