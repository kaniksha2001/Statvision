"""Reusable Streamlit UI building blocks shared across pages.

Keeping these here avoids copy-pasting download buttons, result cards and the
'add to report' affordance into every page, and guarantees a consistent look.
"""
from __future__ import annotations

import uuid
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from statvision.config.styles import get_theme, metric_card
from statvision.core import state
from statvision.core.utils import fmt


def current_template() -> str:
    """Return the Plotly template string matching the active theme.

    Centralised so every chart picks up dark/light mode without each page
    repeating the session-state lookup.
    """
    mode = st.session_state.get("theme", "Light")
    return get_theme(mode)["plotly_template"]


def require_data() -> pd.DataFrame | None:
    """Stop the page early with guidance if no dataset is loaded."""
    if not state.has_data():
        st.info("📂 No dataset loaded yet. Head to **Home** to upload a CSV, "
                "Excel or TSV file, or load the bundled sample dataset.")
        st.stop()
    return state.get_df()


def page_header(title: str, subtitle: str = "", help_text: str = "") -> None:
    """Render a consistent page title with an optional info tooltip."""
    cols = st.columns([0.92, 0.08])
    with cols[0]:
        st.markdown(f"## {title}")
        if subtitle:
            st.markdown(f"<div class='sv-sub'>{subtitle}</div>", unsafe_allow_html=True)
    if help_text:
        with cols[1]:
            st.markdown("&nbsp;")
            st.caption("ℹ️")
            st.session_state.setdefault("_tips", {})
    if help_text:
        with st.expander("ℹ️ About this analysis"):
            st.markdown(help_text)


def metric_row(metrics: dict[str, Any]) -> None:
    """Render a row of metric cards from a label -> value dictionary."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        text = value if isinstance(value, str) else f"{value:,}" if isinstance(value, int) else fmt(value)
        col.markdown(metric_card(label, text), unsafe_allow_html=True)


def show_chart(fig: go.Figure, key: str | None = None,
               report_title: str | None = None) -> None:
    """Render a plotly chart with PNG/HTML download and 'add to report'."""
    key = key or f"chart_{uuid.uuid4().hex[:8]}"
    st.plotly_chart(fig, use_container_width=True, key=key)
    cols = st.columns([0.25, 0.25, 0.5])
    try:
        png = fig.to_image(format="png", width=1000, height=560, scale=2)
        cols[0].download_button("⬇ PNG", png, file_name=f"{key}.png",
                                mime="image/png", key=f"png_{key}")
    except Exception:
        cols[0].caption("PNG export needs `kaleido`")
    html = fig.to_html(include_plotlyjs="cdn")
    cols[1].download_button("⬇ HTML", html, file_name=f"{key}.html",
                            mime="text/html", key=f"html_{key}")
    if report_title:
        if cols[2].button("➕ Add to report", key=f"rep_{key}"):
            state.add_report_item(report_title, "figure", fig)
            st.toast(f"Added '{report_title}' to report")


def show_png(png_bytes: bytes, caption: str = "", report_title: str | None = None,
             key: str | None = None) -> None:
    """Render a Matplotlib/Seaborn PNG with download and report buttons."""
    key = key or f"png_{uuid.uuid4().hex[:8]}"
    st.image(png_bytes, caption=caption, use_container_width=True)
    cols = st.columns([0.3, 0.7])
    cols[0].download_button("⬇ PNG", png_bytes, file_name=f"{key}.png",
                            mime="image/png", key=f"dl_{key}")
    if report_title and cols[1].button("➕ Add to report", key=f"reppng_{key}"):
        state.add_report_item(report_title, "figure", png_bytes)
        st.toast(f"Added '{report_title}' to report")


def show_table(df: pd.DataFrame, report_title: str | None = None,
               key: str | None = None, height: int | None = None) -> None:
    """Render a dataframe with an optional 'add to report' button."""
    kwargs: dict[str, Any] = {"use_container_width": True}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)
    if report_title:
        key = key or f"tbl_{uuid.uuid4().hex[:8]}"
        if st.button("➕ Add to report", key=f"reptbl_{key}"):
            state.add_report_item(report_title, "table", df)
            st.toast(f"Added '{report_title}' to report")


def test_result_card(result: dict, report_title: str | None = None) -> None:
    """Render a standardised hypothesis/inference result card."""
    decision = result.get("decision", "")
    pill = "ok" if "Fail" in decision else "bad"
    st.markdown(
        f"<div class='sv-card'><b>{result.get('test', 'Result')}</b>"
        f"<span class='sv-pill {pill}'>{decision}</span></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    if "h0" in result:
        cols[0].markdown(f"**H₀:** {result['h0']}")
        cols[1].markdown(f"**H₁:** {result['h1']}")
    m = st.columns(3)
    m[0].metric("Statistic", fmt(result.get("statistic")))
    p = result.get("p_value")
    m[1].metric("P-value", "—" if p is None else fmt(p))
    if "critical_value_5pct" in result:
        m[2].metric("Crit. value (5%)", fmt(result["critical_value_5pct"]))
    st.caption(result.get("interpretation", ""))
    if report_title and st.button("➕ Add to report",
                                  key=f"repres_{uuid.uuid4().hex[:8]}"):
        payload = {k: result[k] for k in
                   ("test", "statistic", "p_value", "decision", "interpretation")
                   if k in result}
        state.add_report_item(report_title, "metrics", payload)
        st.toast("Added result to report")
