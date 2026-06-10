"""Hypothesis testing: parametric, non-parametric and categorical tests.

Each function returns a uniform dictionary including the null/alternative
hypotheses so the UI can present a complete, teaching-oriented result card.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from statvision.core.utils import decision_text


def _pack(name: str, h0: str, h1: str, stat: float, p: float, alpha: float,
          extra: dict | None = None) -> dict:
    reject = (not np.isnan(p)) and p < alpha
    result = {
        "test": name,
        "h0": h0,
        "h1": h1,
        "statistic": float(stat),
        "p_value": float(p),
        "alpha": alpha,
        "decision": decision_text(p, alpha),
        "interpretation": (
            "The result is statistically significant; reject the null hypothesis."
            if reject else
            "The result is not statistically significant; insufficient evidence to reject the null."
        ),
    }
    if extra:
        result.update(extra)
    return result


# --- Parametric -------------------------------------------------------------
def one_sample_t(data: np.ndarray, popmean: float, alpha: float = 0.05) -> dict:
    """One-sample t-test against a hypothesised population mean."""
    stat, p = stats.ttest_1samp(data, popmean)
    return _pack("One-Sample t-Test", f"μ = {popmean}", f"μ ≠ {popmean}",
                 stat, p, alpha, {"mean": float(np.mean(data)), "popmean": popmean})


def independent_t(a: np.ndarray, b: np.ndarray, equal_var: bool = False,
                  alpha: float = 0.05) -> dict:
    """Independent two-sample t-test (Welch's by default)."""
    stat, p = stats.ttest_ind(a, b, equal_var=equal_var)
    label = "Independent t-Test" + ("" if equal_var else " (Welch)")
    return _pack(label, "μ₁ = μ₂", "μ₁ ≠ μ₂", stat, p, alpha,
                 {"mean_1": float(np.mean(a)), "mean_2": float(np.mean(b))})


def paired_t(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
    """Paired-samples t-test for repeated measurements."""
    stat, p = stats.ttest_rel(a, b)
    return _pack("Paired t-Test", "μ_d = 0", "μ_d ≠ 0", stat, p, alpha,
                 {"mean_diff": float(np.mean(a - b))})


def one_way_anova(groups: list[np.ndarray], alpha: float = 0.05) -> dict:
    """One-way ANOVA across two or more groups."""
    stat, p = stats.f_oneway(*groups)
    return _pack("One-Way ANOVA", "All group means are equal",
                 "At least one group mean differs", stat, p, alpha,
                 {"n_groups": len(groups)})


# --- Non-parametric ---------------------------------------------------------
def mann_whitney(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
    """Mann-Whitney U test (non-parametric two independent samples)."""
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return _pack("Mann-Whitney U", "Distributions are equal",
                 "Distributions differ", stat, p, alpha)


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
    """Wilcoxon signed-rank test (non-parametric paired samples)."""
    stat, p = stats.wilcoxon(a, b)
    return _pack("Wilcoxon Signed-Rank", "Median difference = 0",
                 "Median difference ≠ 0", stat, p, alpha)


def kruskal_wallis(groups: list[np.ndarray], alpha: float = 0.05) -> dict:
    """Kruskal-Wallis H test (non-parametric one-way ANOVA analogue)."""
    stat, p = stats.kruskal(*groups)
    return _pack("Kruskal-Wallis H", "All group distributions are equal",
                 "At least one distribution differs", stat, p, alpha,
                 {"n_groups": len(groups)})


# --- Categorical ------------------------------------------------------------
def chi_square_independence(df: pd.DataFrame, col_a: str, col_b: str,
                            alpha: float = 0.05) -> dict:
    """Chi-square test of independence between two categorical variables."""
    contingency = pd.crosstab(df[col_a], df[col_b])
    stat, p, dof, expected = stats.chi2_contingency(contingency)
    return _pack(
        "Chi-Square Independence",
        f"'{col_a}' and '{col_b}' are independent",
        f"'{col_a}' and '{col_b}' are associated",
        stat, p, alpha,
        {"dof": int(dof), "contingency": contingency,
         "expected": pd.DataFrame(expected, index=contingency.index,
                                  columns=contingency.columns)},
    )
