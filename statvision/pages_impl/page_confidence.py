"""Confidence Intervals page.

Builds confidence intervals for the mean, a proportion and the variance at a
configurable confidence level (90 / 95 / 99%), reporting the point estimate,
bounds and margin of error.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from statvision.config.settings import CONFIDENCE_LEVELS
from statvision.core import state
from statvision.core.utils import clean_series, numeric_columns
from statvision.inference import confidence_intervals as ci
from statvision.pages_impl.components import (current_template, page_header,
                                              require_data, show_chart)


def _interval_figure(result: dict, tpl: str) -> go.Figure:
    """Draw the estimate as a point with an error bar spanning the interval."""
    est = result["estimate"]
    fig = go.Figure()
    fig.add_scatter(
        x=[est], y=[result["parameter"]], mode="markers",
        marker={"size": 14},
        error_x={"type": "data",
                 "symmetric": False,
                 "array": [result["upper"] - est],
                 "arrayminus": [est - result["lower"]]},
        name="Estimate ± CI")
    fig.update_layout(template=tpl,
                      title=f"{int(result['confidence']*100)}% Confidence Interval",
                      xaxis_title="Value", yaxis_title="")
    return fig


def _report_ci(name: str, result: dict) -> None:
    """Record a CI result in the report as a metrics block."""
    if st.button("➕ Add to report", key=f"ci_rep_{name}"):
        state.add_report_item(
            name, "metrics",
            {"Estimate": result["estimate"], "Lower": result["lower"],
             "Upper": result["upper"], "Margin of error": result["margin_of_error"],
             "Confidence": f"{int(result['confidence']*100)}%"})
        st.toast("Added interval to report")


def render() -> None:
    page_header(
        "Confidence Intervals",
        "Estimate a population parameter with a range at a chosen confidence level.",
        help_text=("A 95% confidence interval means that if the sampling were "
                   "repeated many times, about 95% of the computed intervals would "
                   "contain the true parameter. The **mean** interval uses Student's "
                   "t, the **proportion** uses the Wilson score interval, and the "
                   "**variance** uses the chi-square distribution."),
    )
    df = require_data()
    tpl = current_template()
    num = numeric_columns(df)

    level_label = st.radio("Confidence level", list(CONFIDENCE_LEVELS),
                           index=1, horizontal=True, key="ci_level")
    conf = CONFIDENCE_LEVELS[level_label]

    tab_mean, tab_prop, tab_var = st.tabs(["Mean", "Proportion", "Variance"])

    # --- Mean --------------------------------------------------------------
    with tab_mean:
        if not num:
            st.warning("No numeric columns available.")
        else:
            col = st.selectbox("Variable", num, key="ci_mean_col")
            data = clean_series(df[col])
            if data.size < 2:
                st.warning("Need at least two values.")
            else:
                res = ci.mean_ci(data, confidence=conf)
                m = st.columns(4)
                m[0].metric("Estimate (mean)", f"{res['estimate']:.4g}")
                m[1].metric("Lower bound", f"{res['lower']:.4g}")
                m[2].metric("Upper bound", f"{res['upper']:.4g}")
                m[3].metric("Margin of error", f"{res['margin_of_error']:.4g}")
                show_chart(_interval_figure(res, tpl), key="ci_mean_fig")
                _report_ci(f"Mean CI — {col}", res)

    # --- Proportion --------------------------------------------------------
    with tab_prop:
        st.markdown("Estimate the proportion of values meeting a condition.")
        mode = st.radio("Input", ["From a column", "Enter counts"],
                        horizontal=True, key="ci_prop_mode")
        method = st.selectbox("Method", ["wilson", "wald"], key="ci_prop_method")
        res = None
        if mode == "From a column":
            allcols = df.columns.tolist()
            col = st.selectbox("Column", allcols, key="ci_prop_col")
            series = df[col].dropna()
            uniques = series.unique().tolist()
            if len(uniques) == 0:
                st.warning("Column has no values.")
            else:
                success = st.selectbox("Count rows where value equals",
                                       uniques, key="ci_prop_val")
                successes = int((series == success).sum())
                n = int(series.size)
                st.caption(f"{successes} of {n} rows match.")
                if n > 0:
                    res = ci.proportion_ci(successes, n, confidence=conf,
                                           method=method)
        else:
            c1, c2 = st.columns(2)
            successes = c1.number_input("Successes", 0, 10_000_000, 50, 1,
                                        key="ci_prop_s")
            n = c2.number_input("Sample size n", 1, 10_000_000, 100, 1,
                                key="ci_prop_n")
            if successes <= n:
                res = ci.proportion_ci(int(successes), int(n), confidence=conf,
                                       method=method)
            else:
                st.warning("Successes cannot exceed the sample size.")
        if res:
            m = st.columns(4)
            m[0].metric("Estimate (p̂)", f"{res['estimate']:.4g}")
            m[1].metric("Lower bound", f"{res['lower']:.4g}")
            m[2].metric("Upper bound", f"{res['upper']:.4g}")
            m[3].metric("Margin of error", f"{res['margin_of_error']:.4g}")
            show_chart(_interval_figure(res, tpl), key="ci_prop_fig")
            _report_ci("Proportion CI", res)

    # --- Variance ----------------------------------------------------------
    with tab_var:
        if not num:
            st.warning("No numeric columns available.")
        else:
            col = st.selectbox("Variable", num, key="ci_var_col")
            data = clean_series(df[col])
            if data.size < 2:
                st.warning("Need at least two values.")
            else:
                res = ci.variance_ci(data, confidence=conf)
                m = st.columns(4)
                m[0].metric("Estimate (variance)", f"{res['estimate']:.4g}")
                m[1].metric("Lower bound", f"{res['lower']:.4g}")
                m[2].metric("Upper bound", f"{res['upper']:.4g}")
                m[3].metric("Std dev range",
                            f"{res['std_dev_lower']:.3g} – {res['std_dev_upper']:.3g}")
                show_chart(_interval_figure(res, tpl), key="ci_var_fig")
                _report_ci(f"Variance CI — {col}", res)
