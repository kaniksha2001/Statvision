"""Reporting page.

Lists every result the user has added to the report from other pages, lets
them write conclusions, and exports a consolidated PDF or Excel workbook via
:mod:`statvision.reporting.report`.
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd
import streamlit as st

from statvision.core import state
from statvision.pages_impl.components import page_header
from statvision.reporting import report as report_builder

_KIND_ICON = {"table": "📋", "text": "📝", "figure": "📊", "metrics": "🔢"}


def _preview_item(item: dict) -> None:
    """Show a compact preview of a single accumulated report item."""
    icon = _KIND_ICON.get(item["kind"], "•")
    with st.expander(f"{icon} {item['title']}  ·  {item['kind']}"):
        payload = item["payload"]
        if item["kind"] == "table" and isinstance(payload, pd.DataFrame):
            st.dataframe(payload, use_container_width=True)
        elif item["kind"] == "metrics" and isinstance(payload, dict):
            st.json(payload)
        elif item["kind"] == "text":
            st.write(payload)
        elif item["kind"] == "figure":
            if isinstance(payload, (bytes, bytearray)):
                st.image(payload, use_container_width=True)
            else:  # plotly figure
                st.plotly_chart(payload, use_container_width=True,
                                key=f"rep_prev_{item['title']}")


def render() -> None:
    page_header(
        "Reporting",
        "Assemble your saved results into a downloadable PDF or Excel report.",
        help_text=("Throughout the app, the **➕ Add to report** buttons collect "
                   "tables, charts and test results here. Add a conclusion below, "
                   "then export everything as a polished PDF or a multi-sheet "
                   "Excel workbook."),
    )

    items = st.session_state.get("report_items", [])
    df = state.get_df()
    df_name = st.session_state.get("df_name")

    c1, c2 = st.columns([0.7, 0.3])
    c1.markdown(f"**{len(items)}** item(s) currently in the report.")
    if c2.button("🗑 Clear report", disabled=not items, key="rep_clear"):
        state.clear_report()
        st.rerun()

    if not items:
        st.info("No results added yet. Use the **➕ Add to report** buttons on "
                "the analysis pages to collect tables, charts and test results.")

    for item in items:
        _preview_item(item)

    st.markdown("### Conclusions")
    conclusions = st.text_area(
        "Write your interpretation and conclusions (included in the report)",
        height=160, key="rep_conclusions",
        placeholder="Summarise the key findings, decisions and recommendations…")

    st.markdown("### Export")
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    col_pdf, col_xlsx = st.columns(2)

    with col_pdf:
        if st.button("📄 Generate PDF", key="rep_gen_pdf",
                     use_container_width=True):
            try:
                with st.spinner("Building PDF…"):
                    pdf_bytes = report_builder.build_pdf(
                        items, df, df_name, conclusions)
                st.session_state["_pdf_bytes"] = pdf_bytes
            except Exception as exc:
                st.error(f"Could not build the PDF: {exc}")
        if st.session_state.get("_pdf_bytes"):
            st.download_button(
                "⬇ Download PDF", st.session_state["_pdf_bytes"],
                file_name=f"statvision_report_{stamp}.pdf",
                mime="application/pdf", key="rep_dl_pdf",
                use_container_width=True)

    with col_xlsx:
        if st.button("📊 Generate Excel", key="rep_gen_xlsx",
                     use_container_width=True):
            try:
                with st.spinner("Building workbook…"):
                    xlsx_bytes = report_builder.build_excel(
                        items, df, df_name, conclusions)
                st.session_state["_xlsx_bytes"] = xlsx_bytes
            except Exception as exc:
                st.error(f"Could not build the workbook: {exc}")
        if st.session_state.get("_xlsx_bytes"):
            st.download_button(
                "⬇ Download Excel", st.session_state["_xlsx_bytes"],
                file_name=f"statvision_report_{stamp}.xlsx",
                mime=("application/vnd.openxmlformats-officedocument."
                      "spreadsheetml.sheet"),
                key="rep_dl_xlsx", use_container_width=True)
