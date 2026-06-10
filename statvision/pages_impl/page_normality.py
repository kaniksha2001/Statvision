"""Normality Testing page.

Runs all five normality tests at once via :func:`normality.run_all` and shows
the Q-Q plot, histogram-with-normal-overlay and density curve diagnostics.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from statvision.core.utils import clean_series, numeric_columns
from statvision.inference import normality
from statvision.pages_impl.components import (current_template, page_header,
                                              require_data, show_chart,
                                              show_table, test_result_card)
from statvision.visualizations import charts


def render() -> None:
    page_header(
        "Normality Testing",
        "Assess whether a variable is plausibly drawn from a normal distribution.",
        help_text=("Several tests are reported because each has different power and "
                   "assumptions: **Shapiro-Wilk** is powerful for small samples, "
                   "**Jarque-Bera** and **D'Agostino K²** focus on skewness/kurtosis, "
                   "and **Kolmogorov-Smirnov** / **Anderson-Darling** compare the full "
                   "distribution. A small p-value is evidence *against* normality."),
    )
    df = require_data()
    tpl = current_template()
    num = numeric_columns(df)
    if not num:
        st.warning("No numeric columns available.")
        st.stop()

    c1, c2 = st.columns([0.6, 0.4])
    col = c1.selectbox("Variable", num, key="norm_col")
    alpha = c2.selectbox("Significance α", [0.10, 0.05, 0.01], index=1,
                         key="norm_alpha")
    data = clean_series(df[col])
    if data.size < 8:
        st.warning("Need at least 8 non-missing values for normality testing.")
        st.stop()

    # --- Run every test ----------------------------------------------------
    st.markdown("### Test results")
    try:
        results = normality.run_all(data, alpha=alpha)
    except Exception as exc:
        st.error(f"Normality tests failed: {exc}")
        st.stop()

    for res in results:
        test_result_card(res, report_title=f"{res['test']} — {col}")

    # Compact summary table for quick scanning / report export.
    summary = pd.DataFrame([
        {"Test": r["test"],
         "Statistic": round(r["statistic"], 4),
         "P-value": ("—" if r["p_value"] != r["p_value"]  # NaN check
                     else round(r["p_value"], 4)),
         "Decision": r["decision"]}
        for r in results
    ])
    st.markdown("### Summary")
    show_table(summary, report_title=f"Normality summary — {col}")

    # --- Visual diagnostics ------------------------------------------------
    st.markdown("### Visual diagnostics")
    show_chart(charts.qq_plot(df, col, template=tpl),
               key=f"norm_qq_{col}", report_title=f"Q-Q plot — {col}")
    show_chart(charts.histogram_with_normal(df, col, template=tpl),
               key=f"norm_hist_{col}",
               report_title=f"Histogram + normal — {col}")
    show_chart(charts.density_plot(df, col, template=tpl),
               key=f"norm_density_{col}", report_title=f"Density — {col}")
