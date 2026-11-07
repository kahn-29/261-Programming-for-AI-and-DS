"""Reusable EDA helpers for the Vietnam real-estate notebook."""

from .cleaning import clean_data, prepare_priced_data
from .config import CATEGORY_COLUMNS, EXPECTED_COLUMNS, RANDOM_STATE
from .duplicates import exact_duplicate_mask, soft_duplicate_mask, soft_key_hash
from .flags import (
    flag_above_quantile,
    flag_negative,
    flag_nonpositive,
    flag_outside_quantile_band,
)
from .plotting import boxplot_stats, hist_as_bars
from .reporting import print_exclusion_waterfall, print_filter_waterfall, title_with_n
from .stats import group_median_table, group_summary_table, pairwise_corr
from .text import normalize_text

__all__ = [
    "CATEGORY_COLUMNS",
    "EXPECTED_COLUMNS",
    "RANDOM_STATE",
    "boxplot_stats",
    "clean_data",
    "exact_duplicate_mask",
    "flag_above_quantile",
    "flag_negative",
    "flag_nonpositive",
    "flag_outside_quantile_band",
    "group_median_table",
    "group_summary_table",
    "hist_as_bars",
    "normalize_text",
    "pairwise_corr",
    "prepare_priced_data",
    "print_exclusion_waterfall",
    "print_filter_waterfall",
    "soft_duplicate_mask",
    "soft_key_hash",
    "title_with_n",
]
