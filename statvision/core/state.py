"""Centralised Streamlit session-state management.

A single ``AppState`` accessor wraps ``st.session_state`` so the rest of the app
never pokes at raw keys. It tracks the active dataframe, an undo history of
cleaning operations, accumulated analysis results (for the report) and the
selected theme.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def init_state() -> None:
    """Initialise all session-state keys with sensible defaults (idempotent)."""
    defaults: dict[str, Any] = {
        "df": None,            # active dataframe
        "df_name": None,       # source filename
        "df_history": [],      # list[(label, dataframe)] for undo
        "report_items": [],    # accumulated results for the report builder
        "theme": "Light",
        "page": "home",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def has_data() -> bool:
    """True when a dataframe is loaded."""
    return isinstance(st.session_state.get("df"), pd.DataFrame)


def get_df() -> pd.DataFrame | None:
    """Return the active dataframe (or ``None``)."""
    return st.session_state.get("df")


def set_df(df: pd.DataFrame, name: str | None = None, push_history: bool = True,
           label: str = "") -> None:
    """Replace the active dataframe, optionally snapshotting the previous one."""
    if push_history and has_data():
        prev = st.session_state["df"]
        st.session_state["df_history"].append((label or "edit", prev.copy()))
        # Keep history bounded to avoid unbounded memory growth.
        st.session_state["df_history"] = st.session_state["df_history"][-25:]
    st.session_state["df"] = df
    if name is not None:
        st.session_state["df_name"] = name


def undo() -> bool:
    """Revert to the previous dataframe snapshot. Returns ``True`` if undone."""
    if st.session_state["df_history"]:
        _, prev = st.session_state["df_history"].pop()
        st.session_state["df"] = prev
        return True
    return False


def add_report_item(title: str, kind: str, payload: Any) -> None:
    """Record an analysis result so it can be included in the exported report.

    ``kind`` is one of ``{"table", "text", "figure", "metrics"}`` and ``payload``
    is the corresponding data (DataFrame, str, plotly/matplotlib figure, dict).
    """
    st.session_state["report_items"].append(
        {"title": title, "kind": kind, "payload": payload}
    )


def clear_report() -> None:
    """Empty the accumulated report items."""
    st.session_state["report_items"] = []
