"""Dataset-level constants for the Vietnam real-estate EDA notebook."""

RANDOM_STATE = 42

EXPECTED_COLUMNS = [
    "name",
    "description",
    "property_type_name",
    "province_name",
    "district_name",
    "ward_name",
    "street_name",
    "project_name",
    "price",
    "area",
    "floor_count",
    "frontage_width",
    "house_depth",
    "road_width",
    "bedroom_count",
    "bathroom_count",
    "house_direction",
    "balcony_direction",
    "published_at",
]

CATEGORY_COLUMNS = [
    "property_type_name",
    "province_name",
    "district_name",
    "ward_name",
    "house_direction",
    "balcony_direction",
]

SOFT_KEY_TEXT_COLS = ["name", "province_name", "district_name", "ward_name"]
SOFT_KEY_VALUE_COLS = ["area", "price"]

NONPOSITIVE_FLAG_COLS = ["price", "area"]
NEGATIVE_FLAG_COLS = ["floor_count", "bedroom_count", "bathroom_count"]
EXTREME_UPPER_FLAG_COLS = ["area", "floor_count", "bedroom_count", "bathroom_count"]

# --- Project paths (resolved from this file, so notebooks may run from any folder) ---
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "dataset"            # raw download (git-ignored)
INTERIM_DIR = ROOT_DIR / "data" / "interim"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

RAW_PARQUET = RAW_DIR / "vietnam_real_estates.parquet"
CLEAN_PARQUET = PROCESSED_DIR / "clean_listings.parquet"    # df_clean (all rows + flags)
PRICED_PARQUET = PROCESSED_DIR / "priced_listings.parquet"  # df_priced (valid price & area)
