"""Accented, price-redacted listing text for Vietnamese language models (PhoBERT).

The cohort's ``text`` column is accent-stripped for TF-IDF. This module rebuilds the
same redaction on the original accented text, joined by ``row_id``, without
touching the frozen cohort. Invariant: normalize(text_vi) == cohort text.
"""

import os
from pathlib import Path
import re
import unicodedata

import pandas as pd
import pyarrow.parquet as pq

from .data import MONEY, PROCESSED, RAW, REPORTS, ROOT, digest, normalize, write_json

CLAUSE = re.compile(r"[\n.!?;,]+")              # same delimiters as redact_text
PHONE = re.compile(r"\[phone_number\]", re.IGNORECASE)
NUMERIC_TOKEN = re.compile(r"\b\w*\d\w*\b")
OUTPUT = PROCESSED / "text_vi.parquet"


def redact_accented(value):
    """redact_text, but keeps Vietnamese diacritics and letter case.

    NFKC composes diacritics (the form PhoBERT was trained on) and maps
    compatibility characters (m², styled letters) like the NFKD in normalize().
    Monetary cues are detected on the accent-stripped clause, exactly as before.
    """
    text = " ".join(unicodedata.normalize("NFKC", str(value)).split())
    kept = [clause for clause in CLAUSE.split(text) if not MONEY.search(normalize(clause))]
    text = PHONE.sub(" ", " ".join(kept))
    text = NUMERIC_TOKEN.sub(" ", text)
    return " ".join(text.split())


def load_raw_text(row_ids):
    """Fetch name/description from the verified raw shards by 'shard:position' id."""
    ids = pd.Series(row_ids, dtype=str)
    shard, position = ids.str.split(":", expand=True).T.values
    frames = []
    for name in sorted(set(shard)):
        mask = shard == name
        table = pq.read_table(RAW / f"{name}.parquet", columns=["name", "description"])
        part = table.take(position[mask].astype(int)).to_pandas()
        part.insert(0, "row_id", ids[mask].to_numpy())
        frames.append(part)
    return pd.concat(frames).set_index("row_id").loc[ids].reset_index()


VNCORENLP_URL = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/"
# Word segmentation needs only the jar and the RDRSegmenter files.
VNCORENLP_FILES = ["VnCoreNLP-1.2.jar", "models/wordsegmenter/vi-vocab",
                   "models/wordsegmenter/wordsegmenter.rdr"]


def download_vncorenlp(save_dir):
    """Fetch the segmenter with requests; py_vncorenlp.download_model shells out to
    wget, which macOS lacks, and fails silently before its file moves."""
    import requests
    for name in VNCORENLP_FILES:
        target = save_dir / name
        if target.exists() and target.stat().st_size > 0:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        with requests.get(VNCORENLP_URL + name, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    output.write(block)
        temporary.rename(target)


def segment(texts, save_dir=ROOT / ".cache/vncorenlp"):
    """Word-segment with RDRSegmenter (VnCoreNLP), PhoBERT's pretraining segmenter."""
    import py_vncorenlp
    save_dir = Path(save_dir).resolve()
    download_vncorenlp(save_dir)
    cwd = os.getcwd()
    try:
        model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(save_dir))
    finally:
        os.chdir(cwd)   # py_vncorenlp changes the working directory
    return [" ".join(model.word_segment(text)) if text else "" for text in texts]


def build(with_segmentation=True):
    cohort = pd.read_parquet(PROCESSED / "experiment_cohort.parquet", columns=["row_id", "text"])
    raw = load_raw_text(cohort.row_id)
    assert raw.row_id.tolist() == cohort.row_id.tolist()
    joined = raw[["name", "description"]].fillna("").agg(" . ".join, axis=1)
    text_vi = joined.map(redact_accented)

    same = text_vi.map(normalize).to_numpy() == cohort.text.to_numpy()
    mismatched = cohort.row_id[~same].tolist()
    # Only styled-Unicode rows may differ (old lower() ran before NFKD).
    if len(mismatched) > 5:
        raise ValueError(f"Redaction diverged from the cohort on {len(mismatched)} rows")

    frame = pd.DataFrame({"row_id": cohort.row_id, "text_vi": text_vi})
    if with_segmentation:
        frame["text_vi_seg"] = segment(frame.text_vi.tolist())
    frame.to_parquet(OUTPUT, index=False)
    write_json(REPORTS / "text_vi_manifest.json", {
        "cohort_sha256": digest(PROCESSED / "experiment_cohort.parquet"),
        "text_vi_sha256": digest(OUTPUT), "rows": len(frame),
        "invariant_mismatch_row_ids": mismatched,
        "segmenter": "VnCoreNLP RDRSegmenter" if with_segmentation else None,
    })
    return frame


if __name__ == "__main__":
    result = build()
    print(result.head(3).to_string())
