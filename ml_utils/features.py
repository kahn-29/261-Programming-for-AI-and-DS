"""Feature engineering. Skeleton: fill in as notebook 03 develops.

Keep every function pure (DataFrame in, DataFrame out) so the same code runs for train and test.
Anything that learns from data (target encoding, TF-IDF, scalers) must be fit on the training
fold only, e.g. inside an sklearn Pipeline.
"""

import pandas as pd

# TODO(03): ideas to try
#   - log features: log_area, log_frontage_width, ...
#   - ratios: frontage_width / house_depth, area / floor_count, bathrooms per bedroom
#   - time: publish year/month (only if the split allows it)
#   - geography: target/frequency encoding for district, ward, street; optional geocoding
#   - text flags from `name`/`description` via regex (so do/hong/hem xe hoi/mat tien/noi that)
#   - IMPORTANT: strip money amounts from text before using it, otherwise the label leaks.


def add_basic_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Placeholder. Return a copy with engineered numeric features."""
    raise NotImplementedError("Implement in notebook 03, then move the final version here.")


def strip_price_mentions(text: pd.Series) -> pd.Series:
    """Placeholder. Remove amounts such as '5,2 ty' / '800 trieu' / 'gia 35tr/m2' from text."""
    raise NotImplementedError
