"""Modelling constants. Paths and RANDOM_STATE are shared with eda_utils."""

from eda_utils.config import (  # noqa: F401  (re-exported for notebooks)
    CLEAN_PARQUET,
    FIGURES_DIR,
    INTERIM_DIR,
    MODELS_DIR,
    PRICED_PARQUET,
    PROCESSED_DIR,
    RANDOM_STATE,
)

# Targets. Predict the log scale: asking prices are heavily right-skewed.
TARGET_PRICE = "log_price"            # log10(price)
TARGET_PPM2 = "log_price_per_m2"      # log10(price / area), predict this, then multiply back by area

# Columns that must never be features for the matching target (they ARE the target).
LEAKY_COLUMNS = ["price", "price_per_m2"]

# Group key used to keep near-duplicate listings on the same side of a split.
GROUP_COL = "soft_key"
TIME_COL = "published_at"

N_FOLDS = 5
TEST_FRACTION = 0.15   # latest listings by TIME_COL held out as the final test set
