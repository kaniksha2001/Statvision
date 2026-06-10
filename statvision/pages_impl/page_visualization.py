"""Data Visualization page.

Exposes every interactive chart in :mod:`statvision.visualizations.charts`
plus the normality and outlier diagnostic plots, with dynamic variable
selection and PNG / HTML / PDF download (handled by the shared components).
"""
from __future__ import annotations

import streamlit as st

from statvision.core.utils import categorical_columns, numeric_columns
from statvision.data_exploration.quality import outlier_mask
from statvision.inference.correlation import correlation_matrix
from statvision.pages_impl.components import (current_template, page_header,
                                              require_data, show_chart,
                                              show_png)
from statvision.visualizations import charts


def render() -> None:
    page_header(
        "Data Visualization",
        "Interactive charts and diagnostic plots with dynamic variable selection.",
        help_text=("Every chart is interactive — hover for values, drag to zoom, "
                   "and use the buttons beneath each plot to download a PNG or an "
                   "interactive HTML copy, or to add it to your report."),
    )
    df = require_data()
    tpl = current_template()
    num = numeric_columns(df)
    cat = categorical_columns(df)

    tabs = st.tabs(
        ["Distribution", "Categorical", "Relationship", "Matrix",
         "Normality", "Outliers"]
    )

    # -- Distribution charts (single numeric variable) ----------------------
    with tabs[0]:
        if not num:
            st.warning("No numeric columns to plot.")
        else:
            col = st.selectbox("Variable", num, key="viz_dist_col")
            kind = st.radio(
                "Chart type",
                ["Histogram", "Histogram + Normal curve", "Density (KDE)",
                 "Histogram + KDE", "Box plot"],
                horizontal=True, key="viz_dist_kind",
            )
            if kind in ("Histogram", "Histogram + Normal curve", "Histogram + KDE"):
                bins = st.slider("Bins", 5, 100, 30, key="viz_bins")
            if kind == "Histogram":
                fig = charts.histogram(df, col, bins=bins, template=tpl)
            elif kind == "Histogram + Normal curve":
                fig = charts.histogram_with_normal(df, col, bins=bins, template=tpl)
            elif kind == "Density (KDE)":
                fig = charts.density_plot(df, col, template=tpl)
            elif kind == "Histogram + KDE":
                fig = charts.kde_plot(df, col, template=tpl)
            else:
                fig = charts.box_plot(df, [col], template=tpl)
            show_chart(fig, key=f"viz_dist_{col}", report_title=f"{kind} — {col}")

    # -- Categorical charts -------------------------------------------------
    with tabs[1]:
        if not cat and not num:
            st.warning("No columns to plot.")
        else:
            options = cat + num
            col = st.selectbox("Variable", options, key="viz_cat_col")
            kind = st.radio("Chart type", ["Bar chart", "Pie chart"],
                            horizontal=True, key="viz_cat_kind")
            top = st.slider("Show top N categories", 3, 30, 10, key="viz_cat_top")
            fig = (charts.bar_chart(df, col, top=top, template=tpl)
                   if kind == "Bar chart"
                   else charts.pie_chart(df, col, top=top, template=tpl))
            show_chart(fig, key=f"viz_cat_{col}", report_title=f"{kind} — {col}")

    # -- Relationship charts (two variables) --------------------------------
    with tabs[2]:
        if len(num) < 1:
            st.warning("Need at least one numeric column.")
        else:
            kind = st.radio("Chart type",
                            ["Scatter", "Line", "Area"],
                            horizontal=True, key="viz_rel_kind")
            if kind == "Scatter":
                c1, c2, c3 = st.columns(3)
                x = c1.selectbox("X", num, key="viz_sx")
                y = c2.selectbox("Y", num, index=min(1, len(num) - 1), key="viz_sy")
                color = c3.selectbox("Colour by (optional)", ["—"] + cat,
                                     key="viz_sc")
                trend = st.checkbox("Add OLS trendline", value=True, key="viz_tr")
                fig = charts.scatter_plot(
                    df, x, y, color=None if color == "—" else color,
                    trendline=trend, template=tpl)
                show_chart(fig, key=f"viz_sc_{x}_{y}",
                           report_title=f"Scatter — {y} vs {x}")
            else:
                x = st.selectbox("X axis", num + cat, key="viz_lx")
                ys = st.multiselect("Y series (numeric)", num,
                                    default=num[:1], key="viz_ly")
                if ys:
                    fig = (charts.line_plot(df, x, ys, template=tpl)
                           if kind == "Line"
                           else charts.area_plot(df, x, ys, template=tpl))
                    show_chart(fig, key=f"viz_{kind}_{x}",
                               report_title=f"{kind} plot")

    # -- Matrix charts ------------------------------------------------------
    with tabs[3]:
        if len(num) < 2:
            st.warning("Need at least two numeric columns.")
        else:
            kind = st.radio(
                "Chart type",
                ["Value heatmap", "Correlation heatmap", "Pair plot", "Joint plot"],
                horizontal=True, key="viz_mat_kind")
            cols = st.multiselect("Columns", num, default=num[:min(5, len(num))],
                                  key="viz_mat_cols")
            if kind == "Value heatmap" and cols:
                show_chart(charts.heatmap(df, cols, template=tpl),
                           key="viz_heatmap", report_title="Value heatmap")
            elif kind == "Correlation heatmap" and len(cols) >= 2:
                corr = correlation_matrix(df, columns=cols)
                show_chart(charts.correlation_heatmap(corr, template=tpl),
                           key="viz_corrheat", report_title="Correlation heatmap")
            elif kind == "Pair plot" and len(cols) >= 2:
                hue = st.selectbox("Colour by (optional)", ["—"] + cat,
                                   key="viz_pp_hue")
                with st.spinner("Rendering pair plot…"):
                    png = charts.pair_plot(
                        df, cols, hue=None if hue == "—" else hue)
                show_png(png, caption="Pair plot", report_title="Pair plot",
                         key="viz_pairplot")
            elif kind == "Joint plot" and len(cols) >= 2:
                c1, c2 = st.columns(2)
                x = c1.selectbox("X", cols, key="viz_jx")
                y = c2.selectbox("Y", cols, index=min(1, len(cols) - 1),
                                 key="viz_jy")
                with st.spinner("Rendering joint plot…"):
                    png = charts.joint_plot(df, x, y)
                show_png(png, caption=f"Joint plot — {y} vs {x}",
                         report_title="Joint plot", key="viz_jointplot")
            else:
                st.info("Select the required number of columns above.")

    # -- Normality diagnostics ----------------------------------------------
    with tabs[4]:
        if not num:
            st.warning("No numeric columns available.")
        else:
            col = st.selectbox("Variable", num, key="viz_norm_col")
            kind = st.radio("Plot", ["Q-Q plot", "P-P plot",
                                     "Histogram + Normal curve", "Density"],
                            horizontal=True, key="viz_norm_kind")
            if kind == "Q-Q plot":
                fig = charts.qq_plot(df, col, template=tpl)
            elif kind == "P-P plot":
                fig = charts.pp_plot(df, col, template=tpl)
            elif kind == "Histogram + Normal curve":
                fig = charts.histogram_with_normal(df, col, template=tpl)
            else:
                fig = charts.density_plot(df, col, template=tpl)
            show_chart(fig, key=f"viz_norm_{col}",
                       report_title=f"{kind} — {col}")

    # -- Outlier diagnostics ------------------------------------------------
    with tabs[5]:
        if not num:
            st.warning("No numeric columns available.")
        else:
            method = st.radio("Detection method", ["IQR", "Z-score"],
                              horizontal=True, key="viz_out_method")
            col = st.selectbox("Variable", num, key="viz_out_col")
            mask = outlier_mask(df[col], method=method.lower().replace("-", ""))
            n_out = int(mask.sum())
            st.caption(f"Detected **{n_out}** outlier(s) in `{col}` "
                       f"using the {method} rule.")
            show_chart(charts.box_plot(df, [col], template=tpl),
                       key=f"viz_outbox_{col}",
                       report_title=f"Boxplot outliers — {col}")
            if len(num) >= 2:
                others = [c for c in num if c != col]
                ycol = st.selectbox("Scatter against", others, key="viz_out_y")
                plot_df = df[[col, ycol]].copy()
                plot_df["Outlier"] = mask.map({True: "Outlier", False: "Normal"})
                fig = charts.scatter_plot(plot_df, col, ycol, color="Outlier",
                                          template=tpl)
                show_chart(fig, key=f"viz_outsc_{col}_{ycol}",
                           report_title=f"Scatter outliers — {ycol} vs {col}")
