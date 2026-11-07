"""Plotting helpers that stay matplotlib-friendly for large frames."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def hist_as_bars(ax, values: pd.Series, bins: int = 50, log_x: bool = False, **kwargs):
    """Compute histogram on full series, draw as bar chart on ``ax``."""
    v = values.dropna().to_numpy()
    if log_x:
        v = v[v > 0]
        v = np.log10(v)
    counts, edges = np.histogram(v, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)
    ax.bar(centers, counts, width=width, align="center", **kwargs)
    return counts, edges


def boxplot_stats(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    groups: Sequence,
    *,
    whisker_iqr: float = 1.5,
) -> list[dict]:
    """
    Build matplotlib ``bxp`` stats (no fliers) from full-frame group quantiles.

    Whiskers use ``whisker_iqr * IQR``, clipped to observed min/max.
    """
    stats: list[dict] = []
    for g in groups:
        x = frame.loc[frame[group_col] == g, value_col].dropna().to_numpy()
        if len(x) == 0:
            continue
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        iqr = q3 - q1
        low = max(x.min(), q1 - whisker_iqr * iqr)
        high = min(x.max(), q3 + whisker_iqr * iqr)
        stats.append(
            {
                "label": str(g),
                "whislo": low,
                "q1": q1,
                "med": med,
                "q3": q3,
                "whishi": high,
                "fliers": [],
            }
        )
    return stats
