"""Probability Distributions page.

Iterates the distribution registry so each distribution's parameters become
auto-generated sliders. Supports plotting the PMF/PDF and CDF and overlaying
multiple distributions for comparison — all driven by the registry, so adding a
distribution upstream needs no change here.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from statvision.distributions import distributions as dist
from statvision.pages_impl.components import (current_template, page_header,
                                              show_chart)


def _param_controls(spec: dist.DistSpec, key_prefix: str) -> dict[str, float]:
    """Render a slider for every parameter of ``spec`` and collect the values."""
    values: dict[str, float] = {}
    cols = st.columns(max(1, len(spec.params)))
    for col, p in zip(cols, spec.params):
        if p.integer:
            values[p.name] = col.slider(
                p.label, int(p.min), int(p.max), int(p.default), int(p.step),
                key=f"{key_prefix}_{p.name}")
        else:
            values[p.name] = col.slider(
                p.label, float(p.min), float(p.max), float(p.default),
                float(p.step), key=f"{key_prefix}_{p.name}")
    return values


def _single_distribution(tpl: str) -> None:
    """Explore one distribution: PMF/PDF, CDF and theoretical moments."""
    names = {spec.name: key for key, spec in dist.ALL_DISTRIBUTIONS.items()}
    kind = st.radio("Family", ["Discrete", "Continuous"], horizontal=True,
                    key="dist_family")
    available = {
        spec.name: key for key, spec in dist.ALL_DISTRIBUTIONS.items()
        if spec.kind == kind.lower()
    }
    choice = st.selectbox("Distribution", list(available), key="dist_choice")
    spec = dist.get_distribution(available[choice])
    st.caption(spec.summary)

    values = _param_controls(spec, "dist_single")
    try:
        x = spec.support(values)
        pmf_pdf = spec.pmf_pdf(x, values)
        cdf = spec.cdf(x, values)
        moments = spec.moments(values)
    except Exception as exc:  # invalid parameter combination
        st.error(f"Could not evaluate the distribution: {exc}")
        return

    m = st.columns(4)
    for col, (label, val) in zip(m, moments.items()):
        col.metric(label, f"{val:.4g}" if np.isfinite(val) else "—")

    label = "PMF" if spec.kind == "discrete" else "PDF"
    fig = go.Figure()
    if spec.kind == "discrete":
        fig.add_bar(x=x, y=pmf_pdf, name=label)
    else:
        fig.add_scatter(x=x, y=pmf_pdf, mode="lines", name=label, fill="tozeroy")
    fig.update_layout(template=tpl, title=f"{choice} — {label}",
                      xaxis_title="x", yaxis_title="Probability")
    show_chart(fig, key="dist_pmf", report_title=f"{choice} {label}")

    cfig = go.Figure()
    cfig.add_scatter(x=x, y=cdf, mode="lines", name="CDF",
                     line={"shape": "hv" if spec.kind == "discrete" else "linear"})
    cfig.update_layout(template=tpl, title=f"{choice} — CDF",
                       xaxis_title="x", yaxis_title="Cumulative probability")
    show_chart(cfig, key="dist_cdf", report_title=f"{choice} CDF")


def _compare_distributions(tpl: str) -> None:
    """Overlay several configured distributions on shared PDF and CDF axes."""
    st.markdown("Add two or more distributions and compare their PDF/PMF and CDF.")
    n = st.number_input("Number of distributions to compare", 2, 5, 2, 1,
                        key="dist_cmp_n")
    func = st.radio("Curve", ["PDF / PMF", "CDF"], horizontal=True,
                    key="dist_cmp_func")
    configs = []
    for i in range(int(n)):
        with st.expander(f"Distribution {i + 1}", expanded=i < 2):
            key = st.selectbox(
                "Distribution",
                list(dist.ALL_DISTRIBUTIONS),
                format_func=lambda k: dist.ALL_DISTRIBUTIONS[k].name,
                key=f"dist_cmp_choice_{i}")
            spec = dist.get_distribution(key)
            st.caption(spec.summary)
            values = _param_controls(spec, f"dist_cmp_{i}")
            configs.append((spec, values))

    fig = go.Figure()
    for idx, (spec, values) in enumerate(configs):
        try:
            x = spec.support(values)
            y = (spec.cdf(x, values) if func == "CDF"
                 else spec.pmf_pdf(x, values))
        except Exception as exc:
            st.warning(f"Distribution {idx + 1} ({spec.name}) skipped: {exc}")
            continue
        name = f"{spec.name}"
        if spec.kind == "discrete" and func != "CDF":
            fig.add_scatter(x=x, y=y, mode="markers+lines", name=name)
        else:
            fig.add_scatter(x=x, y=y, mode="lines", name=name)
    fig.update_layout(template=tpl, title=f"Distribution comparison — {func}",
                      xaxis_title="x", yaxis_title="Probability")
    show_chart(fig, key="dist_compare", report_title=f"Distribution comparison ({func})")


def render() -> None:
    page_header(
        "Probability Distributions",
        "Explore discrete and continuous distributions interactively.",
        help_text=("Adjust the parameter sliders to see how each distribution's "
                   "shape, PMF/PDF and CDF respond. The **Compare** tab overlays "
                   "several distributions on the same axes."),
    )
    tpl = current_template()
    tab_single, tab_compare = st.tabs(["Explore", "Compare"])
    with tab_single:
        _single_distribution(tpl)
    with tab_compare:
        _compare_distributions(tpl)
