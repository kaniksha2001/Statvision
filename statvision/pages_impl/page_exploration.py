"""Data Exploration page."""
from __future__ import annotations

import streamlit as st

from statvision.core import state
from statvision.data_exploration import overview as ov
from statvision.data_exploration import quality as q
from statvision.data_exploration.io import to_csv_bytes, to_excel_bytes
from statvision.pages_impl.components import metric_row, page_header, show_table


def render() -> None:
    page_header("Data Exploration",
                "Inspect structure, search & filter, and profile data quality.")
    df = state.get_df()
    if df is None:
        st.info("Load a dataset from **Home** first.")
        st.stop()

    tabs = st.tabs(["Overview", "Inspect", "Quality Report", "Profiling", "Export"])

    # --- Overview ----------------------------------------------------------
    with tabs[0]:
        o = ov.overview(df)
        metric_row({"Rows": o["rows"], "Columns": o["columns"], "Memory": o["memory"]})
        metric_row({"Numeric cols": o["numeric"], "Categorical cols": o["categorical"],
                    "Missing cells": o["total_missing"]})
        st.markdown("#### Column metadata")
        show_table(ov.column_metadata(df), report_title="Column Metadata")

    # --- Inspect -----------------------------------------------------------
    with tabs[1]:
        c1, c2, c3, c4 = st.columns(4)
        search = c1.text_input("🔎 Search", help="Case-insensitive substring match.")
        scope = c2.selectbox("In column", ["All columns"] + list(df.columns))
        sort_by = c3.selectbox("Sort by", ["(none)"] + list(df.columns))
        ascending = c4.radio("Order", ["Ascending", "Descending"],
                             horizontal=True) == "Ascending"
        filtered = ov.filter_dataframe(
            df, search=search, column=scope,
            sort_by=None if sort_by == "(none)" else sort_by, ascending=ascending)
        st.caption(f"Showing {len(filtered):,} of {len(df):,} rows")
        st.dataframe(filtered, use_container_width=True, height=420)

    # --- Quality -----------------------------------------------------------
    with tabs[2]:
        dup = q.duplicate_report(df)
        metric_row({"Duplicate rows": dup["duplicates"],
                    "Duplicate %": f"{dup['duplicate_pct']}%",
                    "Total missing": int(df.isna().sum().sum())})
        st.markdown("#### Missing values")
        show_table(q.missing_report(df), report_title="Missing Values Report")
        st.markdown("#### Outlier summary")
        method = st.radio("Method", ["iqr", "zscore"], horizontal=True,
                          format_func=lambda m: "IQR (Tukey)" if m == "iqr" else "Z-score")
        show_table(q.outlier_summary(df, method=method), report_title="Outlier Summary")

    # --- Profiling ---------------------------------------------------------
    with tabs[3]:
        st.markdown("Generate a consolidated automated profile of the dataset.")
        if st.button("⚙️ Generate profiling report", type="primary"):
            with st.spinner("Profiling dataset…"):
                report = q.profiling_report(df)
            st.success("Profile generated.")
            o = report["overview"]
            metric_row({"Rows": o["rows"], "Columns": o["columns"],
                        "Missing cells": o["total_missing"], "Memory": o["memory"]})
            st.markdown("#### Per-column metadata")
            st.dataframe(report["columns"], use_container_width=True)
            st.markdown("#### Descriptive snapshot")
            st.dataframe(report["describe"], use_container_width=True)
            if st.button("➕ Add full profile to report"):
                state.add_report_item("Column Metadata", "table", report["columns"])
                state.add_report_item("Missing Values", "table", report["missing"])
                state.add_report_item("Outlier Summary", "table", report["outliers"])
                st.toast("Profile added to report")

    # --- Export ------------------------------------------------------------
    with tabs[4]:
        st.markdown("Download the current (possibly cleaned) dataset.")
        c1, c2 = st.columns(2)
        c1.download_button("⬇ Download CSV", to_csv_bytes(df),
                           file_name="statvision_export.csv", mime="text/csv")
        c2.download_button("⬇ Download Excel", to_excel_bytes(df),
                           file_name="statvision_export.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
