"""Goodness-of-Fit page.

Runs the Chi-square, Kolmogorov-Smirnov and Anderson-Darling tests against a
theoretical distribution fitted to the selected variable, and supports the
result with observed-vs-expected, distribution-fit, Q-Q and residual plots.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from statvision.core.utils import clean_series, numeric_columns
from statvision.inference import goodness_of_fit as gof
from statvision.pages_impl.components import (current_template, page_header,
                                              require_data, show_chart,
                                              test_result_card)
from statvision.visualizations import charts


def render() -> None:
    page_header(
        "Goodness of Fit",
        "Test whether a variable follows a chosen theoretical distribution.",
        help_text=("A **small p-value** (below your α) means the data are unlikely "
                   "to have come from the fitted distribution. The Anderson-Darling "
                   "test compares its statistic to a critical value rather than a "
                   "p-value. Distribution parameters are estimated by maximum "
                   "likelihood from your data."),
    )
    df = require_data()
    tpl = current_template()
    num = numeric_columns(df)
    if not num:
        st.warning("No numeric columns available.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    col = c1.selectbox("Variable", num, key="gof_col")
    dist_label = c2.selectbox("Distribution", list(gof.FIT_DISTRIBUTIONS),
                              key="gof_dist")
    alpha = c3.selectbox("Significance α", [0.10, 0.05, 0.01], index=1,
                         key="gof_alpha")
    dist_name = gof.FIT_DISTRIBUTIONS[dist_label]
    data = clean_series(df[col])
    if data.size < 8:
        st.warning("Need at least 8 non-missing values to fit a distribution.")
        st.stop()

    bins = st.slider("Bins (for chi-square & observed/expected)", 5, 40, 10,
                     key="gof_bins")

    # --- Tests -------------------------------------------------------------
    st.markdown("### Test results")
    try:
        ks = gof.ks_gof(data, dist_name, alpha=alpha)
        test_result_card(ks, report_title=f"KS GOF — {col} vs {dist_label}")
    except Exception as exc:
        st.error(f"Kolmogorov-Smirnov test failed: {exc}")

    try:
        binned = gof.binned_observed_expected(data, dist_name, bins=bins)
        chi = gof.chi_square_gof(binned["observed"], binned["expected"],
                                 alpha=alpha)
        test_result_card(chi, report_title=f"Chi-square GOF — {col} vs {dist_label}")
    except Exception as exc:
        st.error(f"Chi-square test failed: {exc}")
        binned = None

    if dist_name in ("norm", "expon"):
        try:
            ad = gof.anderson_darling_gof(data, dist_name)
            test_result_card(ad, report_title=f"Anderson-Darling — {col}")
        except Exception as exc:
            st.error(f"Anderson-Darling test failed: {exc}")
    else:
        st.caption("Anderson-Darling is available for Normal and Exponential fits.")

    # --- Visual support ----------------------------------------------------
    st.markdown("### Visual diagnostics")
    if binned is not None:
        show_chart(
            charts.observed_expected_bar(
                binned["centers"], binned["observed"], binned["expected"],
                template=tpl),
            key=f"gof_oe_{col}", report_title=f"Observed vs Expected — {col}")

        # Distribution-fit overlay: histogram (density) + fitted PDF curve.
        try:
            from scipy import stats as _st
            params = binned["params"]
            scipy_dist = getattr(_st, dist_name)
            xs = np.linspace(data.min(), data.max(), 200)
            pdf = scipy_dist.pdf(xs, *params)
            fig = go.Figure()
            fig.add_histogram(x=data, histnorm="probability density",
                              name="Observed", opacity=0.6, nbinsx=bins)
            fig.add_scatter(x=xs, y=pdf, mode="lines",
                            name=f"Fitted {dist_label}")
            fig.update_layout(template=tpl,
                              title=f"Distribution fit — {col} ~ {dist_label}",
                              xaxis_title=col, yaxis_title="Density")
            show_chart(fig, key=f"gof_fit_{col}",
                       report_title=f"Distribution fit — {col}")
        except Exception as exc:
            st.warning(f"Could not draw the fit overlay: {exc}")

        # Residual plot: observed minus expected per bin.
        residuals = binned["observed"] - binned["expected"]
        show_chart(
            charts.residual_plot(binned["centers"], residuals, template=tpl),
            key=f"gof_resid_{col}", report_title=f"GOF residuals — {col}")

    # Q-Q plot is most meaningful for the normal reference.
    if dist_name == "norm":
        show_chart(charts.qq_plot(df, col, template=tpl),
                   key=f"gof_qq_{col}", report_title=f"Q-Q plot — {col}")
