"""Immutable source acquisition, bounded-memory inspection, and cohort creation."""

import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/vduydong"
PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "reports/experiments"
REVISION = "a9a66ffa985edcf76b4be59ae2c6f5b1db889c38"
DATASET = "vduydong/vietnam-real-estates"
SEED = 42
SAMPLE_SIZE = 60_000
PROTOCOL_VERSION = "1.0"
NUMERIC = ["area", "floor_count", "frontage_width", "house_depth", "road_width",
           "bedroom_count", "bathroom_count"]
CATEGORICAL = ["property_type_name", "province_name", "district_name", "ward_name",
               "street_name", "project_name", "house_direction", "balcony_direction"]
TEXT = ["name", "description"]
REQUIRED = TEXT + CATEGORICAL + ["price", "published_at"] + NUMERIC


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def split_fingerprint(frame):
    """SHA-256 of split membership, identical on every operating system.

    The line terminator is fixed to CRLF because the frozen manifest was first
    computed on Windows, where ``to_csv`` defaults to ``os.linesep``.
    """
    table = frame[["row_id", "group_id", "split"]].to_csv(index=False, lineterminator="\r\n")
    return hashlib.sha256(table.encode()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")


def download_source():
    """Fetch the pinned revision; verify every raw file against its LFS SHA-256."""
    RAW.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}",
        params={"recursive": "true", "expand": "false"}, timeout=60)
    response.raise_for_status()
    files = []
    for item in sorted(response.json(), key=lambda item: item["path"]):
        if not item["path"].endswith(".parquet"):
            continue
        path = RAW / item["path"]
        expected = item["lfs"]["oid"]
        if not path.exists():
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{item['path']}"
            temporary = path.with_suffix(".part")
            with requests.get(url, stream=True, timeout=(30, 180)) as result:
                result.raise_for_status()
                with temporary.open("wb") as output:
                    for block in result.iter_content(1024 * 1024):
                        output.write(block)
            if digest(temporary) != expected:
                raise ValueError(f"Source checksum mismatch: {temporary.name}")
            temporary.rename(path)
        if path.stat().st_size != item["size"] or digest(path) != expected:
            raise ValueError(f"Raw source differs from the pinned revision: {path.name}")
        files.append({"file": path.name, "bytes": item["size"], "sha256": expected})
    if len(files) != 5:
        raise ValueError(f"Expected 5 source shards, received {len(files)}")
    manifest = {"dataset": DATASET, "revision": REVISION, "files": files}
    write_json(RAW / "source.json", manifest)
    return manifest


def normalize(value):
    text = unicodedata.normalize("NFKD", str(value).lower().replace("đ", "d"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.split())


MONEY = re.compile(r"\b(?:gia|ty|ti|trieu|tr|vnd|dong|usd|price|billion|million)\b|[$₫€]")


def redact_text(value):
    """Discard monetary clauses and numeric tokens before vocabulary learning."""
    clauses = re.split(r"[\n.!?;,]+", normalize(value))
    text = " ".join(clause for clause in clauses if not MONEY.search(clause))
    text = re.sub(r"\[phone_number\]", " ", text)
    text = re.sub(r"\b\w*\d\w*\b", " ", text)
    return " ".join(text.split())


def prepare_data():
    """Inspect every source row, sample eligible rows, and freeze a temporal cohort."""
    source = download_source()
    rng = np.random.default_rng(SEED)
    reservoir = None
    missing = pd.Series(0, index=REQUIRED, dtype="int64")
    counts = {"raw_rows": 0, "invalid_price": 0, "invalid_area": 0,
              "invalid_timestamp": 0, "explicit_rental_only_title": 0,
              "eligible_rows": 0}
    schemas, months, types = {}, {}, {}
    earliest, latest = None, None
    for item in source["files"]:
        path = RAW / item["file"]
        parquet = pq.ParquetFile(path)
        schema = {field.name: str(field.type) for field in parquet.schema_arrow}
        if set(schema) != set(REQUIRED):
            raise ValueError(f"Unexpected columns in {path.name}: {schema}")
        schemas[path.name] = schema
        offset = 0
        for batch in parquet.iter_batches(batch_size=8192):
            frame = batch.to_pandas()
            missing += frame[REQUIRED].isna().sum()
            counts["raw_rows"] += len(frame)
            frame["row_id"] = [f"{path.stem}:{i}" for i in range(offset, offset + len(frame))]
            offset += len(frame)
            frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
            dates = pd.to_datetime(frame.published_at, format="ISO8601", errors="coerce", utc=True)
            valid_price = np.isfinite(frame.price) & frame.price.gt(0)
            valid_area = np.isfinite(frame.area) & frame.area.gt(0)
            title = frame.name.fillna("").map(normalize)
            rental = title.str.contains(r"\b(?:cho thue|can thue|for rent)\b", regex=True)
            sale = title.str.contains(r"\b(?:ban|chuyen nhuong|for sale)\b", regex=True)
            rental_only = rental & ~sale
            counts["invalid_price"] += int((~valid_price).sum())
            counts["invalid_area"] += int((~valid_area).sum())
            counts["invalid_timestamp"] += int(dates.isna().sum())
            counts["explicit_rental_only_title"] += int(rental_only.sum())
            valid = valid_price & valid_area & dates.notna() & ~rental_only
            frame = frame.loc[valid].copy()
            frame["published_at"] = dates.loc[valid]
            counts["eligible_rows"] += len(frame)
            if frame.empty:
                continue
            lo, hi = frame.published_at.min(), frame.published_at.max()
            earliest = lo if earliest is None else min(earliest, lo)
            latest = hi if latest is None else max(latest, hi)
            for key, number in frame.published_at.dt.strftime("%Y-%m").value_counts().items():
                months[key] = months.get(key, 0) + int(number)
            for key, number in frame.property_type_name.value_counts().items():
                types[key] = types.get(key, 0) + int(number)
            frame["_priority"] = rng.random(len(frame))
            reservoir = frame if reservoir is None else pd.concat([reservoir, frame], ignore_index=True)
            if len(reservoir) > SAMPLE_SIZE:
                reservoir = reservoir.nsmallest(SAMPLE_SIZE, "_priority").copy()
        print(f"Inspected {path.name}: {offset:,} rows", flush=True)
    frame = reservoir.drop(columns="_priority").copy()
    sampled_rows = len(frame)
    for column in NUMERIC:
        values = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = values.where(np.isfinite(values) & values.ge(0))
    for column in CATEGORICAL:
        frame[column] = frame[column].fillna("__MISSING__").astype(str)
    # Price is deliberately excluded from the identity proxy.
    title = frame.name.fillna("").map(normalize).str.replace(r"\d+(?:[.,]\d+)*", "#", regex=True)
    location = frame[["province_name", "district_name", "ward_name"]].agg("|".join, axis=1)
    key = title + "|" + location + "|" + frame.area.round(2).astype(str)
    frame["group_id"] = key.map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    frame = frame.sort_values(["published_at", "row_id"]).drop_duplicates("group_id", keep="first")
    if len(frame) < 1000:
        raise ValueError("Insufficient eligible distinct listings for the frozen protocol")
    train_end = frame.published_at.iloc[int(.70 * len(frame)) - 1]
    validation_end = frame.published_at.iloc[int(.85 * len(frame)) - 1]
    frame["split"] = np.select(
        [frame.published_at.le(train_end), frame.published_at.le(validation_end)],
        ["train", "validation"], default="test")
    frame["text"] = frame[TEXT].fillna("").agg(" . ".join, axis=1).map(redact_text)
    # Raw titles and descriptions are not retained in the modelling cache.
    frame = frame.drop(columns=TEXT).reset_index(drop=True)
    split_digest = split_fingerprint(frame)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / "experiment_cohort.parquet"
    frame.to_parquet(path, index=False)
    profile = {
        "protocol_version": PROTOCOL_VERSION, "seed": SEED, "sample_budget": SAMPLE_SIZE,
        "source": source, "counts": counts, "raw_schema_by_shard": schemas,
        "raw_null_counts": {key: int(value) for key, value in missing.items()},
        "eligible_month_counts": dict(sorted(months.items())), "eligible_property_type_counts": types,
        "eligible_date_min": str(earliest), "eligible_date_max": str(latest),
        "sampled_rows_before_deduplication": sampled_rows,
        "sample_duplicates_removed": sampled_rows - len(frame), "cohort_rows": len(frame),
        "train_end": str(train_end), "validation_end": str(validation_end),
        "split_counts": {key: int(value) for key, value in frame.split.value_counts().items()},
        "split_sha256": split_digest, "cohort_sha256": digest(path),
        "cohort_price_quantiles_vnd": {str(q): float(frame.price.quantile(q)) for q in [0, .01, .5, .99, 1]},
        "cohort_missing_fraction": {key: float(frame[key].isna().mean()) for key in NUMERIC},
    }
    write_json(REPORTS / "data_manifest.json", profile)
    return frame, profile


def load_data():
    """Load a verified cohort; preparation is explicit and never silently downloads."""
    path = PROCESSED / "experiment_cohort.parquet"
    manifest_path = REPORTS / "data_manifest.json"
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Run python -m experiments.prepare before executing the notebooks.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_version"] != PROTOCOL_VERSION or digest(path) != manifest["cohort_sha256"]:
        raise ValueError("Cohort integrity/protocol mismatch; rerun python -m experiments.prepare.")
    frame = pd.read_parquet(path)
    if split_fingerprint(frame) != manifest["split_sha256"]:
        raise ValueError("Split membership has changed")
    parts = {name: frame.loc[frame.split.eq(name)].reset_index(drop=True)
             for name in ["train", "validation", "test"]}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert set(parts[left].group_id).isdisjoint(parts[right].group_id)
        assert parts[left].published_at.max() < parts[right].published_at.min()
    assert all(len(part) > 1 and np.isfinite(part.price).all() and part.price.gt(0).all()
               for part in parts.values())
    return parts, manifest
