"""Cleaning and analytical-subset builders for listing tables."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from . import config
from .duplicates import exact_duplicate_mask, soft_duplicate_mask
from .flags import (
    flag_above_quantile,
    flag_negative,
    flag_nonpositive,
    flag_outside_quantile_band,
)


def clean_data(
    raw: pd.DataFrame,
    *,
    identity_cols: Sequence[str] | None = None,
    soft_text_cols: Sequence[str] | None = None,
    soft_value_cols: Sequence[str] | None = None,
    datetime_col: str = "published_at",
    nonpositive_cols: Sequence[str] | None = None,
    negative_cols: Sequence[str] | None = None,
    extreme_cols: Sequence[str] | None = None,
    extreme_q: float = 0.999,
) -> pd.DataFrame:
    """
    Shallow-copy ``raw``, parse datetime, add duplicate and validity flags.

    Preserves index and row count. Does not drop rows.
    """
    identity_cols = list(identity_cols or config.EXPECTED_COLUMNS)
    soft_text_cols = list(soft_text_cols or config.SOFT_KEY_TEXT_COLS)
    soft_value_cols = list(soft_value_cols or config.SOFT_KEY_VALUE_COLS)
    nonpositive_cols = list(nonpositive_cols or config.NONPOSITIVE_FLAG_COLS)
    negative_cols = list(negative_cols or config.NEGATIVE_FLAG_COLS)
    extreme_cols = list(extreme_cols or config.EXTREME_UPPER_FLAG_COLS)

    out = raw.copy(deep=False)
    if datetime_col in out.columns:
        out[datetime_col] = pd.to_datetime(out[datetime_col], errors="coerce")

    out["is_exact_dup"] = exact_duplicate_mask(out, subset=identity_cols, keep="first")
    out["is_soft_dup"] = soft_duplicate_mask(
        out, soft_text_cols, soft_value_cols, keep="first"
    )

    for col in nonpositive_cols:
        out[f"flag_nonpositive_{col}"] = flag_nonpositive(out[col])
    for col in negative_cols:
        out[f"flag_negative_{col}"] = flag_negative(out[col])

    for col in extreme_cols:
        positive_only = col == "area"
        out[f"flag_extreme_{col}"] = flag_above_quantile(
            out[col],
            extreme_q,
            positive_only=positive_only,
            nonnegative_only=not positive_only,
        )

    return out


def prepare_priced_data(
    clean: pd.DataFrame,
    *,
    price_col: str = "price",
    area_col: str = "area",
    ratio_col: str = "price_per_m2",
    extreme_flag_col: str = "flag_extreme_price_per_m2",
    low_q: float = 0.001,
    high_q: float = 0.999,
) -> pd.DataFrame:
    """Keep rows with positive price and area; add unit price and extreme flag."""
    mask = clean[price_col].gt(0) & clean[area_col].gt(0)
    priced = clean.loc[mask].copy()
    priced[ratio_col] = priced[price_col] / priced[area_col]
    priced[extreme_flag_col] = flag_outside_quantile_band(
        priced[ratio_col], low_q=low_q, high_q=high_q
    )
    return priced
