"""StatVision — application entry point.

Run with::

    streamlit run app.py

This module wires together the configuration, session state, theming and the
sidebar navigation, then dispatches to the selected page's ``render()``
function. All analytical logic lives in the ``statvision`` package; this file
is intentionally thin.
"""
from __future__ import annotations

import streamlit as st

from statvision.config import settings
from statvision.config.styles import inject_css
from statvision.core import state
from statvision.pages_impl import (page_cleaning, page_confidence,
                                   page_correlation, page_descriptive,
                                   page_distributions, page_exploration,
                                   page_goodness_of_fit, page_home,
                                   page_hypothesis, page_normality,
                                   page_regression, page_reporting,
                                   page_visualization)

# Map each navigation key to the corresponding page's render function.
PAGES = {
    "home": page_home.render,
    "exploration": page_exploration.render,
    "cleaning": page_cleaning.render,
    "descriptive": page_descriptive.render,
    "visualization": page_visualization.render,
    "distributions": page_distributions.render,
    "goodness": page_goodness_of_fit.render,
    "normality": page_normality.render,
    "confidence": page_confidence.render,
    "hypothesis": page_hypothesis.render,
    "correlation": page_correlation.render,
    "regression": page_regression.render,
    "reporting": page_reporting.render,
}


def _sidebar() -> str:
    """Render the sidebar (brand, theme toggle, grouped navigation).

    Returns the navigation key of the page the user has selected.
    """
    with st.sidebar:
        st.markdown(
            f"<div class='sv-brand'>Stat<span>Vision</span></div>"
            f"<div class='sv-sub'>{settings.APP_TAGLINE} · v{settings.APP_VERSION}</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # Theme toggle — persisted in session state and applied via CSS.
        modes = ["Light", "Dark"]
        current = st.session_state.get("theme", "Light")
        choice = st.radio("Appearance", modes, index=modes.index(current),
                          horizontal=True, key="theme_radio")
        if choice != current:
            st.session_state["theme"] = choice
            st.rerun()

        st.divider()

        # Dataset status indicator.
        if state.has_data():
            df = state.get_df()
            name = st.session_state.get("df_name") or "dataset"
            st.caption(f"📂 **{name}** — {df.shape[0]:,} rows × {df.shape[1]} cols")
        else:
            st.caption("📂 No dataset loaded")

        st.divider()

        # Grouped navigation. The active page is tracked in session state so a
        # selection survives reruns triggered elsewhere in the app.
        active = st.session_state.get("page", "home")
        for group, navitems in settings.navigation_groups().items():
            st.markdown(f"<div class='sv-sub'>{group.upper()}</div>",
                        unsafe_allow_html=True)
            for item in navitems:
                label = f"{item.icon}  {item.label}"
                btn_type = "primary" if item.key == active else "secondary"
                if st.button(label, key=f"nav_{item.key}",
                             use_container_width=True, type=btn_type,
                             help=item.help):
                    st.session_state["page"] = item.key
                    st.rerun()
    return st.session_state.get("page", "home")


def main() -> None:
    """Configure the page, initialise state and dispatch to the active page."""
    st.set_page_config(
        page_title=f"{settings.APP_NAME} — {settings.APP_TAGLINE}",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    state.init_state()
    inject_css(st.session_state.get("theme", "Light"))

    page_key = _sidebar()
    render = PAGES.get(page_key, page_home.render)
    try:
        render()
    except Exception as exc:  # last-resort guard so the app never hard-crashes
        st.error("Something went wrong while rendering this page.")
        st.exception(exc)


if __name__ == "__main__":
    main()
