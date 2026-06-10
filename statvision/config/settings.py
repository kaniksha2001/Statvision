"""Application-wide configuration: metadata, navigation map and theming tokens.

Centralising these values keeps the UI layer free of magic strings and makes it
trivial to re-skin or re-order the toolkit without touching business logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

APP_NAME: str = "StatVision"
APP_TAGLINE: str = "Statistical Analysis Toolkit"
APP_VERSION: str = "1.0.0"

# Confidence levels exposed throughout the inference modules.
CONFIDENCE_LEVELS: dict[str, float] = {"90%": 0.90, "95%": 0.95, "99%": 0.99}


@dataclass(frozen=True)
class NavItem:
    """A single sidebar navigation entry."""

    key: str
    label: str
    icon: str
    group: str
    help: str = ""


# Ordered navigation. ``key`` maps to a render function in ``pages_impl``.
NAVIGATION: list[NavItem] = [
    NavItem("home", "Home", "🏠", "Start", "Overview & dataset loading"),
    NavItem("exploration", "Data Exploration", "🔎", "Data",
            "Inspect structure, preview, quality report and profiling"),
    NavItem("cleaning", "Data Cleaning", "🧹", "Data",
            "Handle missing values, duplicates, transforms & feature engineering"),
    NavItem("descriptive", "Descriptive Statistics", "📊", "Analyse",
            "Central tendency, dispersion, position and shape"),
    NavItem("visualization", "Visualization", "📈", "Analyse",
            "Interactive charts and diagnostic plots"),
    NavItem("distributions", "Probability Distributions", "🎲", "Analyse",
            "Explore discrete & continuous distributions interactively"),
    NavItem("goodness", "Goodness of Fit", "🎯", "Inference",
            "Chi-square, Kolmogorov-Smirnov and Anderson-Darling fit tests"),
    NavItem("normality", "Normality Testing", "🔔", "Inference",
            "Shapiro-Wilk, KS, Anderson-Darling, Jarque-Bera, D'Agostino"),
    NavItem("confidence", "Confidence Intervals", "📐", "Inference",
            "Intervals for the mean, proportion and variance"),
    NavItem("hypothesis", "Hypothesis Testing", "⚖️", "Inference",
            "Parametric, non-parametric and categorical tests"),
    NavItem("correlation", "Correlation Analysis", "🔗", "Inference",
            "Pearson, Spearman and Kendall correlation"),
    NavItem("regression", "Regression Analysis", "📉", "Inference",
            "Simple & multiple linear regression with diagnostics"),
    NavItem("reporting", "Reporting", "📄", "Output",
            "Export a consolidated PDF / Excel analysis report"),
]


def navigation_groups() -> dict[str, list[NavItem]]:
    """Return navigation items bucketed by their ``group`` preserving order."""
    groups: dict[str, list[NavItem]] = {}
    for item in NAVIGATION:
        groups.setdefault(item.group, []).append(item)
    return groups
