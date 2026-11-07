"""Printing / labeling helpers for EDA notebooks."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def title_with_n(base: str, frame_name: str, n: int) -> str:
    """Figure/table title that states source frame and row count."""
    return f"{base} [{frame_name}, n={n:,}]"


def print_filter_waterfall(
    stages: Sequence[tuple[str, int, int]],
    *,
    base_n: int,
    header: str = "Exclusion waterfall",
) -> None:
    """
    Print sequential filter stages.

    Each stage is ``(label, rows_remaining, rows_excluded_this_step)``.
    """
    print(header)
    for label, remaining, excluded in stages:
        print(
            f"  {label}: {remaining:,} "
            f"(excluded {excluded:,}, {excluded / base_n:.2%})"
        )


def print_exclusion_waterfall(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    priced: pd.DataFrame,
    *,
    price_col: str = "price",
    area_col: str = "area",
) -> None:
    """Domain wrapper: df → df_clean → priced via positive price then area."""
    n0 = len(raw)
    bad_price = int((~clean[price_col].gt(0)).sum())
    price_ok = clean[clean[price_col] > 0]
    n_price_ok = len(price_ok)
    bad_area = int((~price_ok[area_col].gt(0)).sum())

    print_filter_waterfall(
        [
            ("df rows", n0, 0),
            ("df_clean rows", len(clean), 0),
            ("after exclude missing/nonpositive price", n_price_ok, bad_price),
            (
                "after exclude missing/nonpositive area / df_priced",
                len(priced),
                bad_area,
            ),
        ],
        base_n=n0,
    )
