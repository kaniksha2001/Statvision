"""Data Cleaning page."""
from __future__ import annotations

import streamlit as st

from statvision.cleaning import duplicates as dups
from statvision.cleaning import feature as feat
from statvision.cleaning import missing as miss
from statvision.cleaning import transform as tr
from statvision.core import state
from statvision.core.utils import categorical_columns, numeric_columns
from statvision.pages_impl.components import page_header, require_data


def render() -> None:
    page_header("Data Cleaning",
                "Handle missing values, duplicates, transforms and engineer features. "
                "Every operation is reversible via Undo.")
    df = require_data()

    top = st.columns([0.7, 0.15, 0.15])
    top[0].caption(f"Active dataset: **{st.session_state['df_name']}** — "
                   f"{df.shape[0]:,} × {df.shape[1]}")
    if top[1].button("↩ Undo"):
        st.success("Reverted last change.") if state.undo() else st.warning("Nothing to undo.")
        st.rerun()
    if top[2].button("🔄 Refresh"):
        st.rerun()

    tabs = st.tabs(["Missing Values", "Duplicates", "Transform", "Feature Engineering"])
    num_cols, cat_cols = numeric_columns(df), categorical_columns(df)

    # --- Missing -----------------------------------------------------------
    with tabs[0]:
        st.dataframe(df.isna().sum().rename("Missing").to_frame().T,
                     use_container_width=True)
        action = st.selectbox(
            "Strategy",
            ["Drop rows with missing", "Drop columns over threshold",
             "Mean imputation", "Median imputation", "Mode imputation",
             "Constant value replacement"],
        )
        if action == "Drop columns over threshold":
            thr = st.slider("Drop columns with missing fraction above", 0.0, 1.0, 0.5, 0.05)
        elif action in ("Mean imputation", "Median imputation"):
            cols = st.multiselect("Numeric columns", num_cols, default=num_cols)
        elif action == "Mode imputation":
            cols = st.multiselect("Columns", list(df.columns))
        elif action == "Constant value replacement":
            cols = st.multiselect("Columns", list(df.columns))
            value = st.text_input("Replacement value", "0")
        if st.button("Apply", type="primary", key="apply_missing"):
            try:
                if action == "Drop rows with missing":
                    new = miss.drop_missing_rows(df)
                elif action == "Drop columns over threshold":
                    new = miss.drop_missing_columns(df, thr)
                elif action == "Mean imputation":
                    new = miss.impute_mean(df, cols)
                elif action == "Median imputation":
                    new = miss.impute_median(df, cols)
                elif action == "Mode imputation":
                    new = miss.impute_mode(df, cols)
                else:
                    new = miss.impute_constant(df, cols, value)
                state.set_df(new, label=action)
                st.success(f"Applied: {action}. Now {new.shape[0]:,} × {new.shape[1]}.")
                st.rerun()
            except Exception as exc:
                st.error(f"Operation failed: {exc}")

    # --- Duplicates --------------------------------------------------------
    with tabs[1]:
        subset = st.multiselect("Consider duplicates on columns (blank = all)",
                                list(df.columns))
        subset = subset or None
        n_dup = int(df.duplicated(subset=subset).sum())
        st.metric("Duplicate rows", n_dup)
        c1, c2 = st.columns(2)
        if c1.button("👁 Preview duplicates"):
            st.dataframe(dups.detect_duplicates(df, subset), use_container_width=True)
        if c2.button("🗑 Remove duplicates", type="primary"):
            new = dups.remove_duplicates(df, subset)
            state.set_df(new, label="remove duplicates")
            st.success(f"Removed {n_dup} duplicate rows.")
            st.rerun()

    # --- Transform ---------------------------------------------------------
    with tabs[2]:
        method = st.selectbox("Transformation",
                              ["Standardization (z-score)", "Normalization (L2)",
                               "Min-Max Scaling", "Log Transformation"])
        cols = st.multiselect("Numeric columns", num_cols, default=num_cols[:1])
        if st.button("Apply transform", type="primary", key="apply_tr") and cols:
            try:
                if method.startswith("Standard"):
                    new = tr.standardize(df, cols)
                elif method.startswith("Normal"):
                    new = tr.normalize(df, cols)
                elif method.startswith("Min-Max"):
                    new = tr.min_max_scale(df, cols)
                else:
                    new = tr.log_transform(df, cols)
                state.set_df(new, label=method)
                st.success(f"Added transformed columns via {method}.")
                st.rerun()
            except Exception as exc:
                st.error(f"Transform failed: {exc}")

    # --- Feature engineering ----------------------------------------------
    with tabs[3]:
        op = st.selectbox("Operation",
                          ["Rename column", "Create column (formula)",
                           "Bin column", "Encode categorical"])
        if op == "Rename column":
            old = st.selectbox("Column", list(df.columns))
            new_name = st.text_input("New name", old)
            if st.button("Rename", type="primary"):
                state.set_df(feat.rename_column(df, old, new_name), label="rename")
                st.success(f"Renamed '{old}' → '{new_name}'.")
                st.rerun()
        elif op == "Create column (formula)":
            name = st.text_input("New column name", "new_feature")
            expr = st.text_input("Formula (pandas eval)", "",
                                 help="e.g. annual_income / 12  or  age * 2")
            if st.button("Create", type="primary") and expr:
                try:
                    state.set_df(feat.create_column(df, name, expr), label="create col")
                    st.success(f"Created column '{name}'.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not evaluate formula: {exc}")
        elif op == "Bin column":
            col = st.selectbox("Numeric column", num_cols)
            bins = st.slider("Number of bins", 2, 20, 5)
            strategy = st.radio("Strategy", ["equal_width", "quantile"], horizontal=True)
            if st.button("Bin", type="primary"):
                try:
                    state.set_df(feat.bin_column(df, col, bins, strategy=strategy),
                                 label="bin")
                    st.success(f"Binned '{col}' into {bins} groups.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Binning failed: {exc}")
        else:
            cols = st.multiselect("Categorical columns", cat_cols, default=cat_cols[:1])
            enc = st.radio("Method", ["onehot", "label"], horizontal=True)
            drop_first = st.checkbox("Drop first (one-hot)", value=False)
            if st.button("Encode", type="primary") and cols:
                state.set_df(feat.encode_categorical(df, cols, enc, drop_first),
                             label="encode")
                st.success(f"Encoded {len(cols)} column(s) with {enc}.")
                st.rerun()

    st.divider()
    st.markdown("#### Preview")
    st.dataframe(df.head(12), use_container_width=True)
