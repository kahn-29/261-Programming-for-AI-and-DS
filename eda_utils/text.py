"""Text normalization utilities."""

from __future__ import annotations

import unicodedata

import pandas as pd


def normalize_text(series: pd.Series) -> pd.Series:
    """Lowercase, strip accents (NFKD), map Vietnamese đ→d, collapse whitespace."""
    s = series.fillna("").astype(str).str.lower()
    s = s.map(lambda x: unicodedata.normalize("NFKD", x))
    s = s.map(lambda x: "".join(ch for ch in x if not unicodedata.combining(ch)))
    s = s.str.replace("đ", "d", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s
