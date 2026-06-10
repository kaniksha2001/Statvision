"""Theme tokens and CSS injection helpers for light / dark presentation.

Streamlit's native theming is configured in ``.streamlit/config.toml`` but we
layer additional CSS on top so the app reads like a polished desktop product
rather than a default Streamlit script.
"""
from __future__ import annotations

import streamlit as st

# Colour tokens for each mode. Kept here so charts and CSS share one source.
THEMES: dict[str, dict[str, str]] = {
    "Light": {
        "bg": "#f7f8fa",
        "panel": "#ffffff",
        "text": "#1f2933",
        "muted": "#6b7280",
        "primary": "#2563eb",
        "accent": "#0d9488",
        "border": "#e5e7eb",
        "plotly_template": "plotly_white",
    },
    "Dark": {
        "bg": "#0e1117",
        "panel": "#161b22",
        "text": "#e6edf3",
        "muted": "#9ba4b0",
        "primary": "#3b82f6",
        "accent": "#2dd4bf",
        "border": "#30363d",
        "plotly_template": "plotly_dark",
    },
}


def get_theme(mode: str) -> dict[str, str]:
    """Return the colour token dictionary for the requested mode."""
    return THEMES.get(mode, THEMES["Light"])


def inject_css(mode: str) -> None:
    """Inject global CSS so the chosen theme is applied consistently."""
    t = get_theme(mode)
    st.markdown(
        f"""
        <style>
        :root {{
            --sv-bg: {t['bg']};
            --sv-panel: {t['panel']};
            --sv-text: {t['text']};
            --sv-muted: {t['muted']};
            --sv-primary: {t['primary']};
            --sv-accent: {t['accent']};
            --sv-border: {t['border']};
        }}
        .stApp {{ background-color: var(--sv-bg); }}
        section[data-testid="stSidebar"] {{
            background-color: var(--sv-panel);
            border-right: 1px solid var(--sv-border);
        }}
        h1, h2, h3, h4 {{ color: var(--sv-text); font-weight: 650; }}
        .sv-card {{
            background: var(--sv-panel);
            border: 1px solid var(--sv-border);
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1rem;
        }}
        .sv-metric {{
            background: var(--sv-panel);
            border: 1px solid var(--sv-border);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            text-align: center;
        }}
        .sv-metric .v {{ font-size: 1.6rem; font-weight: 700; color: var(--sv-primary); }}
        .sv-metric .l {{ font-size: 0.8rem; color: var(--sv-muted); text-transform: uppercase;
                         letter-spacing: 0.04em; }}
        .sv-pill {{
            display:inline-block; padding: 2px 10px; border-radius: 999px;
            font-size: 0.75rem; font-weight:600; margin-left: 6px;
        }}
        .sv-pill.ok {{ background: rgba(16,185,129,.15); color:#059669; }}
        .sv-pill.warn {{ background: rgba(245,158,11,.15); color:#d97706; }}
        .sv-pill.bad {{ background: rgba(239,68,68,.15); color:#dc2626; }}
        .sv-brand {{ font-size: 1.45rem; font-weight: 800; color: var(--sv-text); }}
        .sv-brand span {{ color: var(--sv-primary); }}
        .sv-sub {{ color: var(--sv-muted); font-size: 0.82rem; margin-top:-6px; }}
        div[data-testid="stMetricValue"] {{ color: var(--sv-primary); }}
        .stButton>button {{ border-radius: 8px; font-weight: 600; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> str:
    """Return HTML for a compact metric card (used with ``st.markdown``)."""
    return f"<div class='sv-metric'><div class='v'>{value}</div><div class='l'>{label}</div></div>"
