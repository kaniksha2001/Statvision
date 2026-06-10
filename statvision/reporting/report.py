"""Consolidated report generation (PDF & Excel).

The reporting engine consumes a list of *report items* — dictionaries with a
``title``, ``kind`` (``text`` | ``table`` | ``metrics`` | ``figure``) and a
``payload`` — accumulated as the user works, plus a dataset summary, and emits
self-contained PDF or Excel bytes.

PDF rendering uses Matplotlib's :class:`PdfPages` (no external binary needed).
Plotly figures are converted to PNG via kaleido when available; if conversion
fails the figure is skipped gracefully with a placeholder note so report
generation never crashes.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from statvision.config.settings import APP_NAME, APP_VERSION


# --- figure conversion helpers ---------------------------------------------
def _figure_to_png_bytes(payload: Any) -> bytes | None:
    """Best-effort conversion of a report figure payload to PNG bytes.

    Accepts raw PNG bytes (Matplotlib/Seaborn output) or a Plotly figure.
    Returns ``None`` if conversion is not possible.
    """
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    # Plotly figure -> needs kaleido
    try:
        import plotly.graph_objects as go

        if isinstance(payload, go.Figure):
            return payload.to_image(format="png", width=900, height=500, scale=2)
    except Exception:
        return None
    return None


# --- PDF --------------------------------------------------------------------
def build_pdf(report_items: list[dict], df: pd.DataFrame | None,
              df_name: str | None, conclusions: str = "") -> bytes:
    """Render every accumulated item into a multi-page PDF document."""
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        _pdf_cover(pdf, df, df_name)
        if df is not None:
            _pdf_dataset_summary(pdf, df)
        for item in report_items:
            _pdf_item(pdf, item)
        if conclusions.strip():
            _pdf_text_page(pdf, "Conclusions", conclusions)
    buffer.seek(0)
    return buffer.getvalue()


def _new_page(title: str = ""):
    """Create a portrait A4 figure with an optional page title."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.06)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    if title:
        ax.text(0.08, 0.95, title, fontsize=16, fontweight="bold", va="top")
        ax.axhline(0.93, 0.08, 0.92, color="#2563eb", lw=2)
    return fig, ax


def _pdf_cover(pdf: PdfPages, df: pd.DataFrame | None, df_name: str | None) -> None:
    fig, ax = _new_page()
    ax.text(0.5, 0.62, APP_NAME, fontsize=40, fontweight="bold", ha="center",
            color="#2563eb")
    ax.text(0.5, 0.56, "Statistical Analysis Report", fontsize=18, ha="center",
            color="#374151")
    meta = [
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"Dataset: {df_name or 'N/A'}",
        f"Dimensions: {df.shape[0]:,} rows x {df.shape[1]} cols" if df is not None else "",
        f"{APP_NAME} v{APP_VERSION}",
    ]
    ax.text(0.5, 0.40, "\n".join(m for m in meta if m), fontsize=11, ha="center",
            color="#4b5563", linespacing=1.8)
    pdf.savefig(fig)
    plt.close(fig)


def _pdf_dataset_summary(pdf: PdfPages, df: pd.DataFrame) -> None:
    desc = df.describe(include="all").transpose().round(3)
    desc = desc.reset_index().rename(columns={"index": "Column"})
    _pdf_table_pages(pdf, "Dataset Summary", desc)


def _pdf_item(pdf: PdfPages, item: dict) -> None:
    kind, title, payload = item["kind"], item["title"], item["payload"]
    if kind == "text":
        _pdf_text_page(pdf, title, str(payload))
    elif kind == "metrics":
        lines = [f"{k}: {v}" for k, v in payload.items()]
        _pdf_text_page(pdf, title, "\n".join(lines))
    elif kind == "table":
        table = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
        _pdf_table_pages(pdf, title, table.round(4) if not table.empty else table)
    elif kind == "figure":
        _pdf_figure_page(pdf, title, payload)


def _pdf_text_page(pdf: PdfPages, title: str, text: str) -> None:
    fig, ax = _new_page(title)
    ax.text(0.08, 0.88, text, fontsize=10, va="top", family="monospace",
            wrap=True, linespacing=1.5)
    pdf.savefig(fig)
    plt.close(fig)


def _pdf_table_pages(pdf: PdfPages, title: str, table: pd.DataFrame,
                     rows_per_page: int = 28) -> None:
    """Render a dataframe across as many pages as needed."""
    if table.empty:
        _pdf_text_page(pdf, title, "(no data)")
        return
    display = table.copy()
    for col in display.columns:
        display[col] = display[col].astype(str).str.slice(0, 22)
    n = len(display)
    for start in range(0, n, rows_per_page):
        chunk = display.iloc[start:start + rows_per_page]
        fig, ax = _new_page(title if start == 0 else f"{title} (cont.)")
        tbl = ax.table(cellText=chunk.values, colLabels=chunk.columns,
                       loc="upper center", cellLoc="center",
                       bbox=[0.04, 0.04, 0.92, 0.84])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        for (r, _), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2563eb")
                cell.set_text_props(color="white", fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)


def _pdf_figure_page(pdf: PdfPages, title: str, payload: Any) -> None:
    png = _figure_to_png_bytes(payload)
    fig, ax = _new_page(title)
    if png is None:
        ax.text(0.5, 0.5, "(figure could not be rendered)", ha="center",
                color="#9ca3af")
    else:
        import matplotlib.image as mpimg

        img = mpimg.imread(io.BytesIO(png), format="png")
        img_ax = fig.add_axes([0.08, 0.08, 0.84, 0.80])
        img_ax.imshow(img)
        img_ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


# --- Excel ------------------------------------------------------------------
def build_excel(report_items: list[dict], df: pd.DataFrame | None,
                df_name: str | None, conclusions: str = "") -> bytes:
    """Write a multi-sheet Excel workbook summarising the analysis."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # Overview sheet.
        meta = pd.DataFrame(
            {
                "Field": ["Application", "Version", "Generated", "Dataset",
                          "Rows", "Columns"],
                "Value": [APP_NAME, APP_VERSION, f"{datetime.now():%Y-%m-%d %H:%M}",
                          df_name or "N/A",
                          df.shape[0] if df is not None else 0,
                          df.shape[1] if df is not None else 0],
            }
        )
        meta.to_excel(writer, sheet_name="Overview", index=False)

        if df is not None:
            df.describe(include="all").transpose().to_excel(
                writer, sheet_name="Dataset Summary")

        used_names: set[str] = {"Overview", "Dataset Summary"}
        for i, item in enumerate(report_items, start=1):
            sheet = _safe_sheet_name(item["title"], i, used_names)
            if item["kind"] == "table":
                table = (item["payload"] if isinstance(item["payload"], pd.DataFrame)
                         else pd.DataFrame(item["payload"]))
                table.to_excel(writer, sheet_name=sheet)
            elif item["kind"] == "metrics":
                pd.DataFrame(list(item["payload"].items()),
                             columns=["Metric", "Value"]).to_excel(
                    writer, sheet_name=sheet, index=False)
            elif item["kind"] == "text":
                pd.DataFrame({"Result": str(item["payload"]).split("\n")}).to_excel(
                    writer, sheet_name=sheet, index=False)
            # figures are omitted from Excel by design

        if conclusions.strip():
            pd.DataFrame({"Conclusions": conclusions.split("\n")}).to_excel(
                writer, sheet_name="Conclusions", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def _safe_sheet_name(title: str, idx: int, used: set[str]) -> str:
    """Build a unique, Excel-legal (<=31 char) sheet name."""
    base = "".join(c for c in title if c not in "[]:*?/\\")[:25].strip() or "Sheet"
    name = f"{idx:02d}_{base}"[:31]
    while name in used:
        name = f"{name[:28]}_{idx}"
    used.add(name)
    return name
