"""Dataset import / export.

Reading delegates to pandas with light format detection; writing returns raw
bytes so the Streamlit layer can hand them to ``st.download_button`` without
touching the filesystem.
"""
from __future__ import annotations

import io
from typing import BinaryIO

import pandas as pd


def read_dataset(file: BinaryIO, filename: str) -> pd.DataFrame:
    """Read an uploaded file into a dataframe based on its extension.

    Supports ``.csv``, ``.tsv``/``.txt`` and ``.xlsx``/``.xls``. Raises
    ``ValueError`` for anything else so the caller can surface a clean message.
    """
    name = filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith((".tsv", ".txt")):
        return pd.read_csv(file, sep="\t")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    raise ValueError(f"Unsupported file type: {filename}")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a dataframe to CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    """Serialise a dataframe to an in-memory ``.xlsx`` workbook."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "Data")
    return buffer.getvalue()
