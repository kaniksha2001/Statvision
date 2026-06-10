"""Correlation Analysis page.

Pearson, Spearman and Kendall correlation — a single pairwise coefficient with
its significance test, a full correlation matrix with heatmap, and a scatter
matrix (pair plot) for visual inspection.
"""
from __future__ import annotations

import streamlit as st

from statvision.core.utils import numeric_columns
from statvision.inference import correlation as corr
from statvision.pages_impl.components import (current_template, page_header,
                                              require_data, show_chart,
                                              show_png, show_table)
from statvision.visualizations import charts

_METHOD_LABELS = {"Pearson": "pearson", "Spearman": "spearman",
                  "Kendall": "kendall"}


def render() -> None:
    page_header(
        "Correlation Analysis",
        "Measure the strength and direction of association between variables.",
        help_text=("**Pearson** captures linear association, **Spearman** captures "
                   "any monotonic association via ranks, and **Kendall** measures "
                   "ordinal concordance and is robust for small samples. "
                   "Coefficients range from −1 to +1; values near 0 indicate little "
                   "association."),
    )
    df = require_data()
    tpl = current_template()
    num = numeric_columns(df)
    if len(num) < 2:
        st.warning("Need at least two numeric columns for correlation analysis.")
        st.stop()

    method_label = st.radio("Method", list(_METHOD_LABELS), horizontal=True,
                            key="corr_method")
    method = _METHOD_LABELS[method_label]

    tab_pair, tab_matrix, tab_scatter = st.tabs(
        ["Pairwise", "Matrix & heatmap", "Scatter matrix"])

    # --- Single pairwise coefficient --------------------------------------
    with tab_pair:
        c1, c2 = st.columns(2)
        x = c1.selectbox("Variable X", num, key="corr_x")
        y = c2.selectbox("Variable Y", num, index=min(1, len(num) - 1),
                         key="corr_y")
        if x == y:
            st.info("Choose two different variables.")
        else:
            pair = df[[x, y]].apply(lambda s: s).dropna()
            if pair.shape[0] < 3:
                st.warning("Not enough overlapping observations.")
            else:
                res = corr.pairwise_correlation(
                    pair[x].to_numpy(), pair[y].to_numpy(), method=method)
                m = st.columns(3)
                m[0].metric(f"{res['method']} r", f"{res['coefficient']:.4f}")
                m[1].metric("P-value", f"{res['p_value']:.4g}")
                m[2].metric("n", f"{res['n']}")
                st.caption(res["interpretation"])
                show_chart(
                    charts.scatter_plot(pair, x, y, trendline=True, template=tpl),
                    key=f"corr_sc_{x}_{y}",
                    report_title=f"{res['method']} scatter — {y} vs {x}")

    # --- Correlation matrix + heatmap -------------------------------------
    with tab_matrix:
        cols = st.multiselect("Columns", num, default=num[:min(8, len(num))],
                              key="corr_cols")
        if len(cols) >= 2:
            matrix = corr.correlation_matrix(df, method=method, columns=cols)
            st.markdown(f"**{method_label} correlation matrix**")
            show_table(matrix.round(3),
                       report_title=f"{method_label} correlation matrix")
            show_chart(charts.correlation_heatmap(matrix, template=tpl),
                       key="corr_heatmap",
                       report_title=f"{method_label} correlation heatmap")
        else:
            st.info("Select at least two columns.")

    # --- Scatter matrix (pair plot) ---------------------------------------
    with tab_scatter:
        cols = st.multiselect("Columns", num, default=num[:min(4, len(num))],
                              key="corr_pair_cols")
        if len(cols) >= 2:
            with st.spinner("Rendering scatter matrix…"):
                png = charts.pair_plot(df, cols)
            show_png(png, caption="Scatter matrix", report_title="Scatter matrix",
                     key="corr_scatter_matrix")
        else:
            st.info("Select at least two columns.")
