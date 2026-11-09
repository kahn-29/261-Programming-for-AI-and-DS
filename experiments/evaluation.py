"""Consistent original-scale validation, reporting, and artifact integrity checks."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error

from .data import REPORTS, ROOT, digest, write_json

MODEL_NAMES = {
    "dummy": "Dummy Regressor", "median": "Median Price per sqm",
    "ridge": "Ridge Regression", "forest": "Random Forest", "lightgbm": "LightGBM",
    "catboost": "CatBoost", "tfidf": "TF-IDF + Ridge",
    "mlp": "MLP + Embedding", "phobert": "PhoBERT + MLP",
    "fusion": "Tabular + PhoBERT",
}
# The seven classical models that the summary audit requires.
CORE_MODELS = ["dummy", "median", "ridge", "forest", "lightgbm", "catboost", "tfidf"]
PACKAGES = ["numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm", "catboost",
            "matplotlib", "torch", "transformers"]


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: _version(name) for name in PACKAGES}}


def source_hashes():
    return {name: hashlib.sha256((ROOT / "experiments" / name).read_bytes()).hexdigest()
            for name in ["data.py", "estimators.py", "evaluation.py", "text_vi.py", "deep.py", "phobert.py", "fusion.py"]}


def protocol_id(manifest):
    """Identity of the prediction task: same cohort, same split, same protocol version.

    Hardware, operating system, package versions and source hashes are recorded in
    every result for audit, but they do not define the task, so results produced on
    Windows, macOS and Colab stay comparable.
    """
    value = {"protocol_version": manifest["protocol_version"],
             "cohort": manifest["cohort_sha256"], "split": manifest["split_sha256"]}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def is_comparable(record, manifest):
    """Same task as the current manifest. Older records lack protocol_version, so
    they are matched on split membership and source revision instead."""
    return (record.get("split_sha256") == manifest["split_sha256"]
            and record.get("revision") == manifest["source"]["revision"])


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError("Predictions must be finite and have the target shape")
    if not (actual > 0).all():
        raise ValueError("MAPE requires strictly positive targets in this protocol")
    result = {"MAE": mean_absolute_error(actual, predicted),
              "RMSE": root_mean_squared_error(actual, predicted),
              "R2": r2_score(actual, predicted),
              "MAPE": 100 * mean_absolute_percentage_error(actual, predicted)}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("A non-finite evaluation metric was produced")
    return {key: float(value) for key, value in result.items()}


def inverse_log(prediction, training_price):
    """Bound log predictions using training-only support, then return VND."""
    upper = np.log1p(np.asarray(training_price).max())
    return np.expm1(np.clip(np.asarray(prediction), 0.0, upper))


def select_candidate(candidates, fit_candidate, validation_price, predict_candidate):
    """Select by validation MAE in VND; no test data enter this function."""
    records, best_model, best_mae, best_parameters = [], None, np.inf, None
    for parameters in candidates:
        start = time.perf_counter()
        model = fit_candidate(parameters)
        prediction = predict_candidate(model)
        score = metrics(validation_price, prediction)
        record = {**parameters, **score, "fit_seconds": time.perf_counter() - start}
        records.append(record)
        if score["MAE"] < best_mae:
            best_model, best_mae, best_parameters = model, score["MAE"], parameters.copy()
    return best_model, best_parameters, pd.DataFrame(records)


def record_result(key, manifest, parts, validation_prediction, test_prediction,
                  parameters, tuning, extra=None):
    directory = REPORTS / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    results, prediction_files = {}, {}
    for split, prediction in [("validation", validation_prediction), ("test", test_prediction)]:
        actual = parts[split].price.to_numpy()
        results[split] = metrics(actual, prediction)
        path = directory / f"{key}_{split}.parquet"
        pd.DataFrame({"row_id": parts[split].row_id, "actual_vnd": actual,
                      "predicted_vnd": prediction}).to_parquet(path, index=False)
        prediction_files[split] = {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}
    result = {"model": MODEL_NAMES[key], "key": key, "protocol_id": protocol_id(manifest),
              "split_sha256": manifest["split_sha256"], "dataset": manifest["source"]["dataset"],
              "revision": manifest["source"]["revision"], "environment": environment(),
              "source_code": source_hashes(),
              "selected_parameters": parameters, "tuning": tuning.to_dict(orient="records"),
              "metrics": results, "prediction_files": prediction_files, "extra": extra or {}}
    write_json(REPORTS / f"{key}.json", result)
    table = pd.DataFrame(results).T.rename_axis("Split")
    return result, table


def comparison_table(manifest):
    records = []
    for key, name in MODEL_NAMES.items():
        path = REPORTS / f"{key}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if not is_comparable(record, manifest):
            continue
        records.append({"Model": name, **record["metrics"]["test"]})
    return pd.DataFrame(records, columns=["Model", "MAE", "RMSE", "R2", "MAPE"])


def diagnostics(key, actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    rng = np.random.default_rng(42)
    index = rng.choice(len(actual), size=min(3000, len(actual)), replace=False)
    axes[0].scatter(actual[index] / 1e9, predicted[index] / 1e9,
                    s=5, alpha=.25, edgecolors="none")
    positive = np.r_[actual[index], predicted[index]]
    lo = max(float(positive[positive > 0].min()) / 1e9, 1e-9)
    hi = float(positive.max()) / 1e9
    axes[0].plot([lo, hi], [lo, hi], color="black", linewidth=.8)
    axes[0].set(xscale="log", yscale="log", xlabel="Actual price (billion VND; log scale)",
                ylabel="Predicted price (billion VND; log scale)", title="Held-out price predictions")
    residual = (actual - predicted) / 1e9
    low, high = np.quantile(residual, [.01, .99])
    axes[1].hist(residual, bins=np.linspace(low, high, 51), color="#386c8e")
    axes[1].axvline(0, color="black", linewidth=.8)
    axes[1].set(xlabel="Actual minus predicted price (billion VND)", ylabel="Listing count",
                title="Residuals: central 98% display range")
    directory = REPORTS / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{key}_diagnostics.pdf", bbox_inches="tight")
    plt.show()


def importance_plot(key, names, importance, title):
    values = pd.Series(importance, index=names).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    values.plot.barh(ax=ax, color="#386c8e")
    ax.set(xlabel="Importance (model-specific units)", ylabel="Feature", title=title)
    directory = REPORTS / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{key}_importance.pdf", bbox_inches="tight")
    plt.show()
