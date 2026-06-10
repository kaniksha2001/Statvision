"""Home / landing page: data loading and a feature overview."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from statvision.config.settings import APP_NAME, APP_TAGLINE, NAVIGATION
from statvision.core import state
from statvision.data_exploration.io import read_dataset


def _sample_dataset() -> pd.DataFrame:
    """Generate a realistic mixed-type demo dataset for first-run exploration."""
    rng = np.random.default_rng(42)
    n = 300
    region = rng.choice(["North", "South", "East", "West"], n, p=[.3, .25, .25, .2])
    base = {"North": 60, "South": 52, "East": 48, "West": 55}
    age = rng.integers(18, 70, n)
    income = np.array([base[r] for r in region]) * 1000 + age * 320 + rng.normal(0, 9000, n)
    score = 0.0006 * income + rng.normal(50, 12, n)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "age": age,
            "region": region,
            "membership": rng.choice(["Basic", "Plus", "Premium"], n, p=[.5, .3, .2]),
            "annual_income": income.round(2),
            "satisfaction": score.round(2),
            "purchases": rng.poisson(6, n),
            "churned": rng.choice([0, 1], n, p=[.78, .22]),
        }
    )
    # Sprinkle in a little missingness to make cleaning meaningful.
    miss_idx = rng.choice(n, 18, replace=False)
    df.loc[miss_idx, "annual_income"] = np.nan
    return df


def render() -> None:
    st.markdown(
        f"<div class='sv-brand'>📊 {APP_NAME[:4]}<span>{APP_NAME[4:]}</span></div>"
        f"<div class='sv-sub'>{APP_TAGLINE} — an SPSS/Minitab-style workspace</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([0.55, 0.45])
    with left:
        st.markdown("### Load your data")
        upload = st.file_uploader(
            "Upload CSV, Excel (.xlsx) or TSV",
            type=["csv", "xlsx", "xls", "tsv", "txt"],
            help="Your data stays in this session only and is never uploaded elsewhere.",
        )
        if upload is not None:
            try:
                df = read_dataset(upload, upload.name)
                state.set_df(df, name=upload.name, push_history=False)
                st.success(f"Loaded **{upload.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns.")
            except Exception as exc:  # surface a clean message, never a traceback
                st.error(f"Could not read the file: {exc}")

        st.markdown("##### …or start with sample data")
        if st.button("🎲 Load sample dataset", type="primary"):
            state.set_df(_sample_dataset(), name="sample_customers.csv", push_history=False)
            st.success("Sample dataset loaded. Explore the modules from the sidebar.")

    with right:
        st.markdown("### Current dataset")
        if state.has_data():
            df = state.get_df()
            st.markdown(
                f"<div class='sv-card'><b>{st.session_state['df_name']}</b><br>"
                f"{df.shape[0]:,} rows &middot; {df.shape[1]} columns</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(df.head(8), use_container_width=True)
        else:
            st.markdown("<div class='sv-card'>No dataset loaded.</div>",
                        unsafe_allow_html=True)

    st.divider()
    st.markdown("### Modules")
    cols = st.columns(4)
    for i, item in enumerate(n for n in NAVIGATION if n.key != "home"):
        with cols[i % 4]:
            st.markdown(
                f"<div class='sv-card' style='min-height:96px'>{item.icon} "
                f"<b>{item.label}</b><br><span class='sv-sub'>{item.help}</span></div>",
                unsafe_allow_html=True,
            )
