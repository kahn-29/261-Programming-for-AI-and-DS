"""Reusable validity / extreme-value flag builders."""

from __future__ import annotations

import numpy as np
import pandas as pd


def flag_nonpositive(series: pd.Series) -> pd.Series:
    """True where value <= 0; nulls become False."""
    return series.le(0).fillna(False)


def flag_negative(series: pd.Series) -> pd.Series:
    """True where value < 0; nulls become False."""
    return series.lt(0).fillna(False)


def flag_above_quantile(
    series: pd.Series,
    q: float = 0.999,
    *,
    positive_only: bool = False,
    nonnegative_only: bool = False,
) -> pd.Series:
    """True where value exceeds the q-quantile of a filtered reference set."""
    ref = series
    if positive_only:
        ref = series.loc[series > 0]
    elif nonnegative_only:
        ref = series.loc[series >= 0]
    thr = ref.quantile(q) if len(ref) else np.nan
    return series.gt(thr).fillna(False)


def flag_outside_quantile_band(
    series: pd.Series,
    low_q: float = 0.001,
    high_q: float = 0.999,
) -> pd.Series:
    """True where value is outside [low_q, high_q] quantiles of ``series``."""
    lo = series.quantile(low_q)
    hi = series.quantile(high_q)
    return (series.lt(lo) | series.gt(hi)).fillna(False)
