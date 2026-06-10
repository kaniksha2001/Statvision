"""Goodness-of-fit testing against theoretical distributions.

Each test returns a uniform result dictionary (statistic, p-value, decision,
interpretation) so the UI and report layers can render them generically.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from statvision.core.utils import decision_text

# Distributions a user can fit against, mapped to scipy names.
FIT_DISTRIBUTIONS = {
    "Normal": "norm",
    "Exponential": "expon",
    "Uniform": "uniform",
    "Lognormal": "lognorm",
    "Gamma": "gamma",
    "Weibull": "weibull_min",
}


def _fit_params(data: np.ndarray, dist_name: str) -> tuple:
    """Maximum-likelihood-fit the chosen scipy distribution to the data."""
    return getattr(stats, dist_name).fit(data)


def ks_gof(data: np.ndarray, dist_name: str = "norm", alpha: float = 0.05) -> dict:
    """Kolmogorov-Smirnov one-sample GOF test with MLE-fitted parameters."""
    params = _fit_params(data, dist_name)
    statistic, p = stats.kstest(data, dist_name, args=params)
    return {
        "test": "Kolmogorov-Smirnov",
        "statistic": float(statistic),
        "p_value": float(p),
        "params": params,
        "decision": decision_text(p, alpha),
        "interpretation": (
            f"At α={alpha}, the data {'do not appear to follow' if p < alpha else 'are consistent with'} "
            f"the fitted {dist_name} distribution."
        ),
    }


def anderson_darling_gof(data: np.ndarray, dist_name: str = "norm") -> dict:
    """Anderson-Darling GOF (supports norm, expon, logistic, gumbel families).

    The AD test compares the statistic to critical values rather than producing
    a p-value, so the decision is taken at the 5% critical level.
    """
    ad_name = {"norm": "norm", "expon": "expon"}.get(dist_name, "norm")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = stats.anderson(data, dist=ad_name)
    # Critical value index for the 5% significance level.
    levels = list(result.significance_level)
    idx = levels.index(5.0) if 5.0 in levels else len(levels) // 2
    crit = float(result.critical_values[idx])
    reject = result.statistic > crit
    return {
        "test": "Anderson-Darling",
        "statistic": float(result.statistic),
        "p_value": float("nan"),  # AD uses critical values
        "critical_value_5pct": crit,
        "decision": "Reject H₀" if reject else "Fail to reject H₀",
        "interpretation": (
            f"Statistic {result.statistic:.4f} vs 5% critical value {crit:.4f}: "
            f"{'reject' if reject else 'do not reject'} the {ad_name} hypothesis."
        ),
    }


def chi_square_gof(observed: np.ndarray, expected: np.ndarray | None = None,
                   ddof: int = 0, alpha: float = 0.05) -> dict:
    """Chi-square GOF for categorical / binned counts.

    When ``expected`` is omitted a uniform distribution is assumed. Expected
    counts are rescaled to match the observed total to satisfy scipy's
    sum-equality requirement.
    """
    observed = np.asarray(observed, dtype=float)
    if expected is None:
        expected = np.full_like(observed, observed.sum() / observed.size)
    else:
        expected = np.asarray(expected, dtype=float)
        expected = expected * observed.sum() / expected.sum()
    statistic, p = stats.chisquare(observed, expected, ddof=ddof)
    return {
        "test": "Chi-Square GOF",
        "statistic": float(statistic),
        "p_value": float(p),
        "observed": observed,
        "expected": expected,
        "decision": decision_text(p, alpha),
        "interpretation": (
            f"At α={alpha}, the observed counts {'differ from' if p < alpha else 'are consistent with'} "
            "the expected distribution."
        ),
    }


def binned_observed_expected(data: np.ndarray, dist_name: str, bins: int = 10) -> dict:
    """Bin continuous data and compute expected counts under a fitted dist.

    Useful both for the chi-square test and for the observed-vs-expected chart.
    """
    params = _fit_params(data, dist_name)
    dist = getattr(stats, dist_name)
    counts, edges = np.histogram(data, bins=bins)
    cdf = dist.cdf(edges, *params)
    probs = np.diff(cdf)
    probs = probs / probs.sum()
    expected = probs * counts.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    return {"observed": counts.astype(float), "expected": expected,
            "centers": centers, "edges": edges, "params": params}
