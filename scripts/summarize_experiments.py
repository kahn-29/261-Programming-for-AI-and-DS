"""Audit all saved predictions and executed notebooks, then publish comparison tables."""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.data import REPORTS, load_data, digest, write_json
from experiments.evaluation import MODEL_NAMES, comparison_table, metrics, protocol_id


def main():
    parts, manifest = load_data()
    expected_protocol = protocol_id(manifest)
    audit = []
    largest_test_index = int(parts["test"].price.to_numpy().argmax())
    largest_test_price = float(parts["test"].price.max())
    squared_error_shares = []
    for key in MODEL_NAMES:
        record = json.loads((REPORTS / f"{key}.json").read_text(encoding="utf-8"))
        assert record["protocol_id"] == expected_protocol
        assert record["split_sha256"] == manifest["split_sha256"]
        for split in ["validation", "test"]:
            info = record["prediction_files"][split]
            path = ROOT / info["path"]
            assert digest(path) == info["sha256"]
            frame = pd.read_parquet(path)
            assert list(frame.row_id) == list(parts[split].row_id)
            np.testing.assert_array_equal(frame.actual_vnd, parts[split].price)
            checked = metrics(frame.actual_vnd, frame.predicted_vnd)
            if split == "test":
                squared_errors = np.square(frame.actual_vnd.to_numpy() - frame.predicted_vnd.to_numpy())
                squared_error_shares.append(float(squared_errors[largest_test_index] / squared_errors.sum()))
            for name, value in checked.items():
                np.testing.assert_allclose(value, record["metrics"][split][name], rtol=1e-12)
        audit.append({"model": MODEL_NAMES[key], "prediction_and_metric_audit": "passed"})
    notebook_status = []
    for path in sorted((ROOT / "models").glob("0*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        counts, figures, tables = [], 0, 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            assert cell.execution_count is not None, f"Unexecuted cell: {path.name}"
            counts.append(cell.execution_count)
            for output in cell.outputs:
                assert output.output_type != "error", f"Notebook error: {path.name}"
                figures += "image/png" in output.get("data", {})
                tables += "text/html" in output.get("data", {})
        assert counts == list(range(1, len(counts) + 1)), path.name
        assert figures >= 1 and tables >= 1, path.name
        notebook_status.append({"notebook": path.name, "code_cells": len(counts),
                                "embedded_figures": figures, "rendered_tables": tables,
                                "status": "passed"})
    assert len(notebook_status) == 7
    table = comparison_table(manifest)
    assert len(table) == 7
    assert np.isfinite(table[["MAE", "RMSE", "R2", "MAPE"]].to_numpy()).all()
    table.to_csv(REPORTS / "comparison.csv", index=False, float_format="%.12g")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.barh(table.Model, table.MAE / 1e9, color="#386c8e")
    ax.invert_yaxis()
    ax.set(xlabel="Test MAE (billion VND)", title="Shared temporal holdout: original-scale error")
    fig.savefig(REPORTS / "comparison.pdf", bbox_inches="tight")
    fig.savefig(REPORTS / "comparison.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    rows = ["| Model | MAE (VND) | RMSE (VND) | R² | MAPE (%) |",
            "|---|---:|---:|---:|---:|"]
    for row in table.itertuples(index=False):
        rows.append(f"| {row.Model} | {row.MAE:,.0f} | {row.RMSE:,.0f} | {row.R2:.4f} | {row.MAPE:,.2f} |")
    counts = manifest["counts"]
    text = f'''# Property asking-price experiment summary

Source: [vduydong/vietnam-real-estates](https://huggingface.co/datasets/vduydong/vietnam-real-estates),
revision `{manifest['source']['revision']}`. All five Parquet files were verified against their
published LFS SHA-256 checksums. Original files are unchanged and excluded from Git.

## Data and experimental protocol

- Inspected source rows: {counts['raw_rows']:,}.
- Invalid/nonpositive prices: {counts['invalid_price']:,}; invalid/nonpositive areas: {counts['invalid_area']:,};
  invalid timestamps: {counts['invalid_timestamp']:,}; explicit rental-only titles: {counts['explicit_rental_only_title']:,}.
  Exclusion counts overlap and must not be added.
- Eligible rows: {counts['eligible_rows']:,}. A seed-42 uniform sample of {manifest['sampled_rows_before_deduplication']:,}
  eligible rows was drawn across all five shards before modelling, independent of price magnitude.
- Removed sampled identity-proxy duplicates: {manifest['sample_duplicates_removed']:,}; final cohort: {manifest['cohort_rows']:,}.
- Training/validation/test sizes: {manifest['split_counts']['train']:,} / {manifest['split_counts']['validation']:,} / {manifest['split_counts']['test']:,}.
- Training end: {manifest['train_end']}; validation end: {manifest['validation_end']}.
  Offset-free timestamps are assigned UTC for ordering; their true timezone is unknown.
- Split fingerprint: `{manifest['split_sha256']}`.
- Every preprocessing fit uses training observations. Validation MAE in VND selects candidates;
  boosting early stopping monitors validation log-RMSE. Test labels do not select candidates.
- Nonbaseline estimators fit log1p prices, inverse-transform predictions, and bound extrapolation
  by the maximum training price. Baselines fit original-scale targets. Models are not refitted
  on validation data. All reported metrics use complete original-scale partitions.

## Observed test results

{chr(10).join(rows)}

MAE and RMSE are in VND; R² is unitless; MAPE is a percentage. All targets are positive,
but small or incorrectly extracted prices can disproportionately affect MAPE. No tail observations
were trimmed for scoring. These descriptive estimates do not establish statistical significance.

The largest recorded test price is **{largest_test_price:,.0f} VND**. That single listing
accounts for **{100 * min(squared_error_shares):.2f}%–{100 * max(squared_error_shares):.2f}%** of
the total squared prediction error across the seven fitted models. This observed concentration
limits the interpretability of aggregate RMSE and R². Its label is not externally verified;
the protocol preserves it rather than revising the test set after observing predictions.

![Original-scale MAE comparison](comparison.png)

## Validation

All seven notebooks executed from beginning to end, contain real outputs, rendered HTML tables,
and embedded diagnostic figures, and share the same split fingerprint. Saved prediction row IDs,
source target values, checksums, and recomputed metrics were audited for both validation and test.
The audit passed for every model. `validation_audit.json` contains the machine-readable checks;
`execution_status.json` records actual execution times. The controlled unit tests cover train-only
statistics, unseen-group fallbacks, unseen category handling, TF-IDF vocabulary isolation,
price-text redaction, original-scale metrics, and temporal/entity separation.

## Limitations and interpretation

The experiments use a bounded 60,000-row sampling budget because available RAM at preparation
was approximately 3.4 GiB; they are not full-corpus training runs. The target is a recorded asking
price rather than a verified sale. The source includes multiple property types and may include
undetected rentals, extraction errors, implausible numeric values, and extreme prices. No
retrospective price correction or target-dependent trimming was performed. A title/location/area
proxy, not a verified property identifier, controls repeats within the sample. Reposts with
different wording may remain. Text redaction removes monetary clauses and digit-bearing tokens
but cannot remove all price-related semantics. All geographic areas are pooled; spatial transfer
and repeated temporal backtesting remain unmeasured. Tree importances are associative and
model-specific. These limitations should accompany any paper reporting the results.

## Reproduction

See [EXPERIMENTS.md](../../EXPERIMENTS.md) for commands, dependencies, and model definitions.
Raw data, modelling caches, and prediction files remain local; compact metrics and the executed
notebooks are versioned. Recreate the local predictions by executing the notebooks.
'''
    (REPORTS / "summary.md").write_text(text, encoding="utf-8")
    write_json(REPORTS / "validation_audit.json", {
        "protocol_id": expected_protocol, "split_sha256": manifest["split_sha256"],
        "model_audit": audit, "notebook_audit": notebook_status,
        "finite_metrics": True, "consistent_split": True,
        "largest_recorded_test_price_vnd": largest_test_price,
        "largest_price_listing_squared_error_share_range": [min(squared_error_shares), max(squared_error_shares)],
    })
    print(table.to_string(index=False))
    print("PASS: seven complete notebooks and all saved predictions audited.")


if __name__ == "__main__":
    main()
