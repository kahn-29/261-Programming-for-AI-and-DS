"""Read existing preprocessed data and share one deterministic training split."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "reports/experiments"
SEED = 42
NUMERIC = ["area", "floor_count", "frontage_width", "house_depth", "road_width",
           "bedroom_count", "bathroom_count"]
CATEGORICAL = ["property_type_name", "province_name", "district_name", "ward_name",
               "street_name", "project_name", "house_direction", "balcony_direction"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")


def default_data_path():
    """Prefer the EDA output; retain support for the existing processed cache."""
    configured = os.environ.get("EXPERIMENT_DATA_PATH")
    if configured:
        return Path(configured)
    eda_output = PROCESSED / "priced_listings.parquet"
    if eda_output.exists():
        return eda_output
    cache = PROCESSED / "experiment_cohort.parquet"
    if cache.exists():
        return cache
    return eda_output


def load_data(path=None):
    """Use every supplied row, preserve existing splits, and never prepare raw data."""
    path = Path(path) if path is not None else default_data_path()
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Preprocessed data not found: {path}. Supply the output from notebooks/ "
            "using DATA_PATH or EXPERIMENT_DATA_PATH. Model notebooks do not run EDA.")
    frame = pd.read_parquet(path).reset_index(drop=True)
    missing = sorted(set(NUMERIC + CATEGORICAL + ["price"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Preprocessed input is missing model columns: {missing}")
    for column in NUMERIC + ["price"]:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"{column} must already be numeric in the preprocessed input")
        if np.isinf(frame[column].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"{column} contains infinity; fix it in the preprocessing notebook")
    if frame.price.isna().any() or not frame.price.gt(0).all():
        raise ValueError("Preprocessed prices must be finite and positive; no rows are filtered here")
    if frame.area.isna().any() or not frame.area.gt(0).all():
        raise ValueError("Preprocessed areas must be finite and positive; no rows are filtered here")
    for column in CATEGORICAL:
        values = frame[column].astype(object)
        frame[column] = values.where(values.notna(), np.nan)
    if "row_id" not in frame:
        frame["row_id"] = [f"row:{index}" for index in range(len(frame))]
    if frame.row_id.isna().any() or not frame.row_id.is_unique:
        raise ValueError("Input row_id values must be unique and nonmissing")
    if "split" in frame:
        if frame.split.isna().any() or set(frame.split.unique()) != {"train", "validation", "test"}:
            raise ValueError("Existing split must contain train, validation, and test for every row")
        split_method = "existing split column (unchanged)"
    elif "published_at" in frame:
        dates = pd.to_datetime(frame.published_at, format="ISO8601", errors="coerce", utc=True)
        if dates.isna().any():
            raise ValueError("Preprocessed publication dates must be valid before temporal splitting")
        ordered = dates.sort_values(kind="stable")
        train_end = ordered.iloc[int(.70 * len(frame)) - 1]
        validation_end = ordered.iloc[int(.85 * len(frame)) - 1]
        frame["split"] = np.select([dates.le(train_end), dates.le(validation_end)],
                                    ["train", "validation"], default="test")
        split_method = "chronological 70/15/15; no rows removed"
    else:
        indices = np.arange(len(frame))
        _, held_out = train_test_split(indices, test_size=.30, random_state=SEED)
        validation_index, test_index = train_test_split(held_out, test_size=.50, random_state=SEED)
        frame["split"] = "train"
        frame.loc[validation_index, "split"] = "validation"
        frame.loc[test_index, "split"] = "test"
        split_method = "random 70/15/15, seed 42; no timestamp column"
    parts = {name: frame.loc[frame.split.eq(name)].reset_index(drop=True)
             for name in ["train", "validation", "test"]}
    if any(len(part) < 2 for part in parts.values()):
        raise ValueError("Each split needs at least two rows; supply a usable upstream split")
    if "group_id" in frame:
        if frame.group_id.isna().any():
            raise ValueError("Provided group_id values must be nonmissing")
        for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            if not set(parts[left].group_id).isdisjoint(parts[right].group_id):
                raise ValueError("Input entity groups overlap splits; resolve this upstream")
    membership = frame[["row_id", "split"]].to_csv(index=False, lineterminator="\n")
    try:
        input_path = path.relative_to(ROOT).as_posix()
    except ValueError:
        input_path = path.name
    manifest = {
        "input_path": input_path, "input_sha256": digest(path), "input_rows": len(frame),
        "seed": SEED, "split_method": split_method,
        "split_counts": {name: len(part) for name, part in parts.items()},
        "split_sha256": hashlib.sha256(membership.encode()).hexdigest(),
    }
    return parts, manifest
