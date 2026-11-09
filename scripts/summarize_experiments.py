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
    print("PASS: seven complete notebooks and all saved predictions audited.")


if __name__ == "__main__":
    main()
