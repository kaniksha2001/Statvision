"""Feature engineering: renaming, derived columns, binning and encoding."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rename_column(df: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    """Rename a single column."""
    return df.rename(columns={old: new})


def create_column(df: pd.DataFrame, name: str, expression: str) -> pd.DataFrame:
    """Create a new column from a pandas ``eval`` expression.

    The expression is evaluated against the dataframe namespace, e.g.
    ``"price * quantity"``. Restricted to pandas/numpy operators by ``eval``.
    """
    out = df.copy()
    out[name] = out.eval(expression, engine="python")
    return out


def bin_column(df: pd.DataFrame, column: str, bins: int = 5,
               labels: list[str] | None = None, strategy: str = "equal_width",
               suffix: str = "_binned") -> pd.DataFrame:
    """Discretise a numeric column.

    ``strategy`` is ``"equal_width"`` (``pd.cut``) or ``"quantile"``
    (``pd.qcut``). Returns a new categorical column named ``column + suffix``.
    """
    out = df.copy()
    series = pd.to_numeric(out[column], errors="coerce")
    if strategy == "quantile":
        out[column + suffix] = pd.qcut(series, q=bins, labels=labels, duplicates="drop")
    else:
        out[column + suffix] = pd.cut(series, bins=bins, labels=labels)
    return out


def encode_categorical(df: pd.DataFrame, columns: list[str], method: str = "onehot",
                       drop_first: bool = False) -> pd.DataFrame:
    """Encode categorical columns.

    ``method`` ``"onehot"`` expands to indicator columns; ``"label"`` maps each
    category to an integer code in a new ``_code`` column.
    """
    out = df.copy()
    if method == "label":
        for col in columns:
            out[col + "_code"] = out[col].astype("category").cat.codes
        return out
    return pd.get_dummies(out, columns=columns, drop_first=drop_first)
