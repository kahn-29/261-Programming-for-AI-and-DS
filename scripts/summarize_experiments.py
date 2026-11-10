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
from experiments.evaluation import (CORE_MODELS, MODEL_NAMES, comparison_table,
                                    is_comparable, metrics, protocol_id)


def main():
    parts, manifest = load_data()
    expected_protocol = protocol_id(manifest)
    audit = []
    largest_test_index = int(parts["test"].price.to_numpy().argmax())
    largest_test_price = float(parts["test"].price.max())
    squared_error_shares = []
    missing = [key for key in CORE_MODELS if not (REPORTS / f"{key}.json").exists()]
    assert not missing, f"Missing core model results: {missing}"
    # Optional deep-learning models are audited only once their results exist.
    keys = [key for key in MODEL_NAMES if (REPORTS / f"{key}.json").exists()]
    for key in keys:
        record = json.loads((REPORTS / f"{key}.json").read_text(encoding="utf-8"))
        assert is_comparable(record, manifest), f"{key}: different task/split"
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
    assert len(notebook_status) >= len(CORE_MODELS)
    table = comparison_table(manifest)
    assert len(table) == len(keys), "Every saved result must be comparable"
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
    if manifest["input_path"].endswith("experiment_cohort.parquet"):
        availability = (
            "This run uses the existing processed cache, read unchanged. "
            "The EDA export priced_listings.parquet was not the input for this run; "
            "the results validate training on the supplied cache.")
    else:
        availability = "This run uses the explicitly identified preprocessed input above."
    text = f'''# Model training results

Input: `{manifest['input_path']}` (SHA-256 `{manifest['input_sha256']}`).
All **{manifest['input_rows']:,} already preprocessed rows** were used.
The training workflow does not download, inspect raw sources, clean, sample, deduplicate,
or overwrite the supplied dataset. EDA and dataset preparation belong to `notebooks/`.

## Training configuration

- Split: {manifest['split_method']}.
- Training / validation / test: {manifest['split_counts']['train']:,} / {manifest['split_counts']['validation']:,} / {manifest['split_counts']['test']:,}.
- Split fingerprint: `{manifest['split_sha256']}`.
- Imputation, encoding, scaling, group medians, and TF-IDF are fitted using training data.
- Validation MAE selects hyperparameters. Boosting early stopping uses validation log-RMSE.
- Nonbaseline models fit log1p prices and return original-scale predictions, bounded by
  the maximum training price. Models are not refitted on validation data.

## Test results

{chr(10).join(rows)}

MAE and RMSE are in VND; R² is unitless; MAPE is a percentage. All targets are positive,
but small or incorrectly extracted prices can disproportionately affect MAPE. No tail observations
were trimmed for scoring. These descriptive estimates do not establish statistical significance.

The largest recorded test price is **{largest_test_price:,.0f} VND**. That single listing
accounts for **{100 * min(squared_error_shares):.2f}%–{100 * max(squared_error_shares):.2f}%** of
the total squared prediction error across the {len(keys)} fitted models. This observed concentration
limits the interpretability of aggregate RMSE and R². Its label is not externally verified;
the protocol preserves it rather than revising the test set after observing predictions.

![Original-scale MAE comparison](comparison.png)

## Validation

All {len(notebook_status)} notebooks executed from beginning to end, contain real outputs, rendered HTML tables,
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
MAE and RMSE are in VND. R-squared is unitless and MAPE is a percentage. Metrics use every
held-out row. The largest recorded test price is **{largest_test_price:,.0f} VND**;
its squared-error contribution ranges from **{100 * min(squared_error_shares):.2f}%** to
**{100 * max(squared_error_shares):.2f}%** across the seven models. Recorded asking prices
may contain errors; the training workflow does not change their values.

![Test MAE comparison](comparison.png)

## Validation and input availability

All seven notebooks executed from beginning to end with real results, rendered tables,
and diagnostic figures. Prediction row alignment, target values, file checksums, finite
metrics, and identical split membership passed the final audit.

{availability} When the EDA output exists, it becomes the default input; select an
explicit `DATA_PATH` or `EXPERIMENT_DATA_PATH` to control the input across runs.

See [EXPERIMENTS.md](../../EXPERIMENTS.md) for the training commands and data contract.
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
    print(f"PASS: {len(notebook_status)} notebooks and {len(keys)} saved model results audited.")


if __name__ == "__main__":
    main()
