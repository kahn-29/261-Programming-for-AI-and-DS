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
