"""Train/test splitting that avoids duplicate leakage and respects time."""

import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import GROUP_COL, N_FOLDS, TEST_FRACTION, TIME_COL


def time_holdout(frame: pd.DataFrame, test_fraction: float = TEST_FRACTION, time_col: str = TIME_COL):
    """Split by time: the latest `test_fraction` of listings become the final test set.

    Returns (train_dev, test). Do all tuning on train_dev only; touch test once at the end.
    """
    ts = pd.to_datetime(frame[time_col], errors="coerce")
    cutoff = ts.quantile(1 - test_fraction)
    is_test = ts >= cutoff
    return frame[~is_test], frame[is_test]


def group_kfold(frame: pd.DataFrame, n_splits: int = N_FOLDS, group_col: str = GROUP_COL):
    """Yield (train_idx, valid_idx) positions where a soft-duplicate group never spans both sides."""
    return GroupKFold(n_splits=n_splits).split(frame, groups=frame[group_col])
