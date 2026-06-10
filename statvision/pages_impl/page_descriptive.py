"""Descriptive Statistics page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from statvision.core.utils import clean_series, numeric_columns
from statvision.pages_impl.components import (page_header, require_data,
                                              show_table)
from statvision.statistics import descriptive as d


def render() -> None:
    page_header(
        "Descriptive Statistics",
        "Central tendency, dispersion, position and shape.",
        help_text=("**Skewness** measures asymmetry (0 = symmetric). **Kurtosis** "
                   "here is *excess* kurtosis (0 = normal tails). **CV** is the "
                   "standard deviation relative to the mean, useful for comparing "
                   "variability across differently-scaled variables."),
    )
    df = require_data()
    num_cols = numeric_columns(df)
    if not num_cols:
        st.warning("No numeric columns available for descriptive statistics.")
        st.stop()

    tab_single, tab_summary = st.tabs(["Single variable", "Summary table"])

    with tab_single:
        col = st.selectbox("Variable", num_cols)
        data = clean_series(df[col])
        if data.size == 0:
            st.warning("Selected column has no numeric data.")
            st.stop()
        measures = d.full_describe(data)

        st.markdown("##### Central tendency & dispersion")
        c = st.columns(4)
        c[0].metric("Mean", f"{measures['Mean']:.4g}")
        c[1].metric("Median", f"{measures['Median']:.4g}")
        c[2].metric("Mode", f"{measures['Mode']:.4g}")
        c[3].metric("Trimmed mean", f"{measures['Trimmed Mean (10%)']:.4g}")
        c = st.columns(4)
        c[0].metric("Std dev", f"{measures['Std Dev']:.4g}")
        c[1].metric("Variance", f"{measures['Variance']:.4g}")
        c[2].metric("IQR", f"{measures['IQR']:.4g}")
        c[3].metric("Range", f"{measures['Range']:.4g}")
        c = st.columns(4)
        c[0].metric("Skewness", f"{measures['Skewness']:.4g}")
        c[1].metric("Kurtosis", f"{measures['Kurtosis']:.4g}")
        c[2].metric("CV", f"{measures['Coef. of Variation']:.4g}")
        c[3].metric("Count", f"{int(data.size)}")

        st.markdown("##### Position — quartiles, deciles & percentiles")
        pos = pd.DataFrame({
            "Statistic": list(d.quartiles(data)) + list(d.deciles(data)) + list(d.percentiles(data)),
            "Value": list(d.quartiles(data).values()) + list(d.deciles(data).values())
                     + list(d.percentiles(data).values()),
        })
        show_table(pos, report_title=f"Position statistics — {col}")

    with tab_summary:
        cols = st.multiselect("Variables", num_cols, default=num_cols)
        if cols:
            table = d.summary_table(df, cols)
            show_table(table.round(4), report_title="Summary Statistics")
            st.download_button(
                "⬇ Download summary (CSV)",
                table.to_csv(index=False).encode("utf-8"),
                file_name="summary_statistics.csv", mime="text/csv")
