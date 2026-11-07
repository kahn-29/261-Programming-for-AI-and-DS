"""Duplicate detection helpers."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from . import config
from .text import normalize_text


def soft_key_hash(
    frame: pd.DataFrame,
    text_cols: Sequence[str] | None = None,
    value_cols: Sequence[str] | None = None,
) -> pd.Series:
    """Hash rows on normalized text columns plus raw value columns."""
    text_cols = list(text_cols or config.SOFT_KEY_TEXT_COLS)
    value_cols = list(value_cols or config.SOFT_KEY_VALUE_COLS)
    parts: dict[str, pd.Series] = {}
    for col in text_cols:
        parts[col] = normalize_text(frame[col].astype("string"))
    for col in value_cols:
        parts[col] = frame[col]
    key_df = pd.DataFrame(parts, index=frame.index)
    return pd.util.hash_pandas_object(key_df, index=False)


def exact_duplicate_mask(
    frame: pd.DataFrame,
    subset: Sequence[str] | None = None,
    keep: str | bool = "first",
) -> pd.Series:
    """Boolean mask of exact duplicates over ``subset`` (default: all columns)."""
    return frame.duplicated(subset=list(subset) if subset is not None else None, keep=keep)


def soft_duplicate_mask(
    frame: pd.DataFrame,
    text_cols: Sequence[str] | None = None,
    value_cols: Sequence[str] | None = None,
    keep: str | bool = "first",
) -> pd.Series:
    """Boolean mask of soft-key duplicates."""
    return soft_key_hash(frame, text_cols, value_cols).duplicated(keep=keep)
