"""Tabular aggregation and association helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def group_summary_table(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    *,
    min_n: int = 100,
    top_n: int | None = None,
    agg: str = "median",
) -> pd.DataFrame:
    """Group-level size + aggregate; keep groups with ``n >= min_n``."""
    g = (
        frame.groupby(group_col, observed=True)[value_col]
        .agg(n="size", **{agg: agg})
        .reset_index()
    )
    g = g[g["n"] >= min_n].sort_values("n", ascending=False)
    if top_n is not None:
        g = g.head(top_n)
    return g


def group_median_table(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    min_n: int = 100,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias for median group summaries."""
    return group_summary_table(
        frame, group_col, value_col, min_n=min_n, top_n=top_n, agg="median"
    )


def pairwise_corr(
    frame: pd.DataFrame,
    cols: Sequence[str],
    method: str = "pearson",
    min_n: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise correlation with pairwise deletion.

    Returns ``(corr_matrix, n_matrix)``. Pairs with n < min_n get NaN corr.
    """
    cols = list(cols)
    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    nmat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            sub = frame[[a, b]].dropna()
            n = len(sub)
            nmat.loc[a, b] = n
            if n < min_n:
                mat.loc[a, b] = np.nan
            elif i == j:
                mat.loc[a, b] = 1.0
            else:
                mat.loc[a, b] = sub[a].corr(sub[b], method=method)
    return mat, nmat
