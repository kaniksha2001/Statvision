"""Regression Analysis page.

Simple and multiple ordinary-least-squares regression backed by statsmodels.
Reports R², adjusted R², the coefficient table and full residual diagnostics
(regression line, residuals-vs-fitted and predicted-vs-actual plots).
"""
from __future__ import annotations

import streamlit as st

from statvision.core import state
from statvision.core.utils import numeric_columns
from statvision.inference import regression as reg
from statvision.pages_impl.components import (current_template, page_header,
                                              require_data, show_chart,
                                              show_table)
from statvision.visualizations import charts


def render() -> None:
    page_header(
        "Regression Analysis",
        "Model a numeric outcome as a linear function of one or more predictors.",
        help_text=("**R²** is the share of variance in the outcome explained by the "
                   "model; **adjusted R²** penalises extra predictors. Each "
                   "coefficient's p-value tests whether that predictor adds "
                   "explanatory power. Inspect the residual plots to check the "
                   "linearity and constant-variance assumptions."),
    )
    df = require_data()
    tpl = current_template()
    num = numeric_columns(df)
    if len(num) < 2:
        st.warning("Need at least two numeric columns (one outcome, one predictor).")
        st.stop()

    c1, c2 = st.columns(2)
    target = c1.selectbox("Outcome (dependent variable)", num, key="reg_target")
    predictor_options = [c for c in num if c != target]
    predictors = c2.multiselect(
        "Predictors (independent variables)", predictor_options,
        default=predictor_options[:1], key="reg_predictors")

    if not predictors:
        st.info("Select at least one predictor.")
        st.stop()

    try:
        result = reg.fit_ols(df, target, predictors)
    except Exception as exc:
        st.error(f"Could not fit the model: {exc}")
        st.stop()

    kind = "Simple linear" if len(predictors) == 1 else "Multiple linear"
    st.markdown(f"### {kind} regression")
    st.code(result["equation"], language="text")

    m = st.columns(4)
    m[0].metric("R²", f"{result['r_squared']:.4f}")
    m[1].metric("Adjusted R²", f"{result['adj_r_squared']:.4f}")
    m[2].metric("F p-value", f"{result['f_pvalue']:.4g}")
    m[3].metric("Observations", f"{result['n_obs']}")

    st.markdown("#### Coefficients")
    show_table(result["coefficients"].round(4),
               report_title=f"Regression coefficients — {target}")

    if st.button("➕ Add model summary to report", key="reg_add_summary"):
        state.add_report_item(
            f"Regression — {target} ~ {', '.join(predictors)}", "metrics",
            {"Equation": result["equation"],
             "R²": round(result["r_squared"], 4),
             "Adjusted R²": round(result["adj_r_squared"], 4),
             "F p-value": f"{result['f_pvalue']:.4g}",
             "Observations": result["n_obs"]})
        st.toast("Added model summary to report")

    # --- Diagnostics -------------------------------------------------------
    st.markdown("#### Diagnostic plots")
    if len(predictors) == 1:
        p = predictors[0]
        clean = df[[target, p]].apply(
            lambda s: s).dropna()
        slope = float(result["model"].params[p])
        intercept = float(result["model"].params["const"])
        show_chart(
            charts.regression_line(
                clean[p].to_numpy(), clean[target].to_numpy(),
                slope, intercept, xlabel=p, ylabel=target, template=tpl),
            key="reg_line", report_title=f"Regression line — {target} vs {p}")

    show_chart(
        charts.residual_plot(result["fitted"], result["residuals"], template=tpl),
        key="reg_resid", report_title=f"Residuals vs fitted — {target}")
    show_chart(
        charts.predicted_vs_actual(result["actual"], result["fitted"],
                                   template=tpl),
        key="reg_pva", report_title=f"Predicted vs actual — {target}")

    with st.expander("Full statsmodels summary"):
        st.text(result["summary_text"])
