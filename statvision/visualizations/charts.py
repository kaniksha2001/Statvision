"""Charting layer.

Interactive charts return Plotly figures (zoom/pan/hover for free); the few
plots Plotly handles poorly (pair plot, joint plot) fall back to
Matplotlib/Seaborn. A single ``THEME_TEMPLATE`` switch keeps everything aligned
with the active light/dark theme.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # headless backend for server-side rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats

from statvision.core.utils import clean_series


def _layout(fig: go.Figure, template: str, title: str = "") -> go.Figure:
    """Apply consistent theming to a plotly figure."""
    fig.update_layout(
        template=template,
        title=title,
        margin=dict(l=40, r=20, t=50, b=40),
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# --- Univariate distribution charts ----------------------------------------
def histogram(df: pd.DataFrame, column: str, bins: int = 30,
              template: str = "plotly_white") -> go.Figure:
    """Frequency histogram of a numeric column."""
    fig = px.histogram(df, x=column, nbins=bins, opacity=0.85)
    return _layout(fig, template, f"Histogram of {column}")


def density_plot(df: pd.DataFrame, column: str,
                 template: str = "plotly_white") -> go.Figure:
    """KDE density curve via figure_factory distplot (curve only)."""
    data = clean_series(df[column])
    fig = ff.create_distplot([data], [column], show_hist=False, show_rug=False)
    return _layout(fig, template, f"Density (KDE) of {column}")


def histogram_with_normal(df: pd.DataFrame, column: str, bins: int = 30,
                          template: str = "plotly_white") -> go.Figure:
    """Histogram (density-normalised) with an overlaid fitted normal curve."""
    data = clean_series(df[column])
    mu, sd = data.mean(), data.std(ddof=1)
    fig = go.Figure()
    fig.add_histogram(x=data, histnorm="probability density", nbinsx=bins,
                      name="Observed", opacity=0.7)
    xs = np.linspace(data.min(), data.max(), 200)
    fig.add_scatter(x=xs, y=stats.norm.pdf(xs, mu, sd), mode="lines",
                    name=f"Normal(μ={mu:.2f}, σ={sd:.2f})")
    return _layout(fig, template, f"{column} with Normal Overlay")


def kde_plot(df: pd.DataFrame, column: str,
             template: str = "plotly_white") -> go.Figure:
    """Histogram + KDE combination."""
    data = clean_series(df[column])
    fig = ff.create_distplot([data], [column], show_rug=False)
    return _layout(fig, template, f"Histogram + KDE of {column}")


def box_plot(df: pd.DataFrame, columns: list[str], group: str | None = None,
             template: str = "plotly_white") -> go.Figure:
    """Box plot for one or more numeric columns, optionally grouped."""
    if group:
        fig = px.box(df, x=group, y=columns[0], points="outliers")
    else:
        melted = df[columns].melt(var_name="Variable", value_name="Value")
        fig = px.box(melted, x="Variable", y="Value", points="outliers")
    return _layout(fig, template, "Box Plot")


# --- Categorical / relationship charts --------------------------------------
def bar_chart(df: pd.DataFrame, column: str, top: int = 20,
              template: str = "plotly_white") -> go.Figure:
    """Bar chart of value counts for a (usually categorical) column."""
    counts = df[column].value_counts().head(top).reset_index()
    counts.columns = [column, "Count"]
    fig = px.bar(counts, x=column, y="Count")
    return _layout(fig, template, f"Bar Chart of {column}")


def pie_chart(df: pd.DataFrame, column: str, top: int = 10,
              template: str = "plotly_white") -> go.Figure:
    """Pie chart of category proportions."""
    counts = df[column].value_counts().head(top).reset_index()
    counts.columns = [column, "Count"]
    fig = px.pie(counts, names=column, values="Count", hole=0.3)
    return _layout(fig, template, f"Composition of {column}")


def scatter_plot(df: pd.DataFrame, x: str, y: str, color: str | None = None,
                 trendline: bool = False, template: str = "plotly_white") -> go.Figure:
    """Scatter plot with optional colour grouping and OLS trendline."""
    fig = px.scatter(df, x=x, y=y, color=color,
                     trendline="ols" if trendline else None)
    return _layout(fig, template, f"{y} vs {x}")


def line_plot(df: pd.DataFrame, x: str, y: list[str],
              template: str = "plotly_white") -> go.Figure:
    """Line plot of one or more series against a shared x."""
    fig = px.line(df, x=x, y=y)
    return _layout(fig, template, "Line Plot")


def area_plot(df: pd.DataFrame, x: str, y: list[str],
              template: str = "plotly_white") -> go.Figure:
    """Stacked area plot."""
    fig = px.area(df, x=x, y=y)
    return _layout(fig, template, "Area Plot")


# --- Matrix / correlation charts --------------------------------------------
def heatmap(df: pd.DataFrame, columns: list[str],
            template: str = "plotly_white") -> go.Figure:
    """Heatmap of raw values (z-scored per column for comparability)."""
    sub = df[columns].apply(pd.to_numeric, errors="coerce")
    z = (sub - sub.mean()) / sub.std(ddof=0)
    fig = px.imshow(z.T, aspect="auto", color_continuous_scale="RdBu_r",
                    origin="lower")
    return _layout(fig, template, "Value Heatmap (z-scored)")


def correlation_heatmap(corr: pd.DataFrame,
                        template: str = "plotly_white") -> go.Figure:
    """Heatmap of a correlation matrix with annotated coefficients."""
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    return _layout(fig, template, "Correlation Heatmap")


# --- Normality diagnostics --------------------------------------------------
def qq_plot(df: pd.DataFrame, column: str,
            template: str = "plotly_white") -> go.Figure:
    """Quantile-quantile plot against the standard normal."""
    data = clean_series(df[column])
    (osm, osr), (slope, intercept, _) = stats.probplot(data, dist="norm")
    fig = go.Figure()
    fig.add_scatter(x=osm, y=osr, mode="markers", name="Sample quantiles")
    fig.add_scatter(x=osm, y=slope * osm + intercept, mode="lines",
                    name="Reference line")
    fig.update_xaxes(title="Theoretical quantiles")
    fig.update_yaxes(title="Ordered values")
    return _layout(fig, template, f"Q-Q Plot of {column}")


def pp_plot(df: pd.DataFrame, column: str,
            template: str = "plotly_white") -> go.Figure:
    """Probability-probability plot against the fitted normal."""
    data = np.sort(clean_series(df[column]))
    mu, sd = data.mean(), data.std(ddof=1)
    empirical = np.arange(1, len(data) + 1) / (len(data) + 1)
    theoretical = stats.norm.cdf(data, mu, sd)
    fig = go.Figure()
    fig.add_scatter(x=theoretical, y=empirical, mode="markers", name="P-P")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Reference")
    fig.update_xaxes(title="Theoretical CDF")
    fig.update_yaxes(title="Empirical CDF")
    return _layout(fig, template, f"P-P Plot of {column}")


def residual_plot(fitted: np.ndarray, residuals: np.ndarray,
                  template: str = "plotly_white") -> go.Figure:
    """Residuals vs fitted values with a zero reference line."""
    fig = go.Figure()
    fig.add_scatter(x=fitted, y=residuals, mode="markers", name="Residuals")
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_xaxes(title="Fitted values")
    fig.update_yaxes(title="Residuals")
    return _layout(fig, template, "Residuals vs Fitted")


def predicted_vs_actual(actual: np.ndarray, predicted: np.ndarray,
                        template: str = "plotly_white") -> go.Figure:
    """Predicted vs actual scatter with the identity line."""
    fig = go.Figure()
    fig.add_scatter(x=actual, y=predicted, mode="markers", name="Observations")
    lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect fit",
                    line_dash="dash")
    fig.update_xaxes(title="Actual")
    fig.update_yaxes(title="Predicted")
    return _layout(fig, template, "Predicted vs Actual")


def regression_line(x: np.ndarray, y: np.ndarray, slope: float, intercept: float,
                    xlabel: str, ylabel: str,
                    template: str = "plotly_white") -> go.Figure:
    """Scatter of data with the fitted regression line overlaid."""
    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="markers", name="Data")
    xs = np.linspace(x.min(), x.max(), 100)
    fig.add_scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Fit")
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=ylabel)
    return _layout(fig, template, "Regression Line")


def observed_expected_bar(centers: np.ndarray, observed: np.ndarray,
                          expected: np.ndarray,
                          template: str = "plotly_white") -> go.Figure:
    """Grouped bar chart comparing observed and expected (fit) counts."""
    fig = go.Figure()
    fig.add_bar(x=centers, y=observed, name="Observed")
    fig.add_scatter(x=centers, y=expected, mode="lines+markers", name="Expected (fit)")
    return _layout(fig, template, "Observed vs Expected")


# --- Matplotlib / seaborn fallbacks (returned as PNG bytes) -----------------
def pair_plot(df: pd.DataFrame, columns: list[str], hue: str | None = None,
              dark: bool = False) -> bytes:
    """Seaborn pair plot rendered to PNG bytes."""
    sns.set_theme(style="darkgrid" if dark else "whitegrid")
    cols = columns + ([hue] if hue and hue not in columns else [])
    grid = sns.pairplot(df[cols].dropna(), hue=hue, corner=False, height=2.0)
    return _fig_to_png(grid.figure)


def joint_plot(df: pd.DataFrame, x: str, y: str, kind: str = "scatter",
               dark: bool = False) -> bytes:
    """Seaborn joint plot (scatter/kde/hex/reg) rendered to PNG bytes."""
    sns.set_theme(style="darkgrid" if dark else "whitegrid")
    grid = sns.jointplot(data=df, x=x, y=y, kind=kind, height=6)
    return _fig_to_png(grid.figure)


def _fig_to_png(fig) -> bytes:
    """Serialise a matplotlib figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
