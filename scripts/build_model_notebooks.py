"""Generate concise training notebooks using already preprocessed data."""

from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def markdown(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


SETUP = r'''
from pathlib import Path
import os
import sys

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / "experiments").is_dir() and (path / "requirements.txt").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from experiments.data import load_data, default_data_path, SEED, NUMERIC, CATEGORICAL
from experiments.evaluation import (
    select_candidate, inverse_log, record_result, diagnostics,
    importance_plot, comparison_table,
)
from experiments.estimators import (
    structured_preprocessor, native_features, catboost_features,
    text_pipeline, text_inputs, MedianPricePerSqm,
)

np.random.seed(SEED)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
DATA_PATH = default_data_path()  # Replace with another preprocessed Parquet path if needed.
'''

INPUT = r'''
parts, manifest = load_data(DATA_PATH)
train, validation, test = (parts[name] for name in ["train", "validation", "test"])
y_train = train.price.to_numpy()
y_validation = validation.price.to_numpy()
log_y_train = np.log1p(y_train)
log_y_validation = np.log1p(y_validation)
display(pd.Series({"Input": manifest["input_path"], "Seed": SEED,
                   "Split": manifest["split_method"],
                   **manifest["split_counts"]}, name="Training settings").to_frame())
'''

SPECS = [
    dict(key="dummy", filename="01_baseline_dummy_regressor.ipynb", title="Dummy Regressor",
         objective="Establish a feature-independent reference using the arithmetic mean training asking price.",
         method="The mean strategy minimizes training squared error among constant predictions. It provides a reference for assessing the added predictive value of structured and textual attributes.",
         implementation='''
from sklearn.dummy import DummyRegressor

def fit_candidate(parameters):
    estimator = DummyRegressor(**parameters)
    estimator.fit(np.zeros((len(train), 1)), y_train)
    return estimator

def predict_frame(estimator, frame):
    return estimator.predict(np.zeros((len(frame), 1)))
''', grid='candidates = [{"strategy": "mean"}]',
         discussion="A constant predictor cannot represent geography, property type, or area. Its mean is sensitive to large recorded prices. Its role is a reference, not a deployable valuation model."),
    dict(key="median", filename="02_baseline_median_price.ipynb", title="Median Price per Square Metre",
         objective="Evaluate a domain-informed baseline that multiplies training-group median unit prices by property area.",
         method='''Training unit price is `price / area`. The primary key is `(province-qualified district, property_type_name)`; adding province disambiguates districts with identical names. Unseen keys fall back to `(province, property type)`, then property type, then the global training median unit price. Missing keys are explicit categories. At inference, invalid areas fall back to the median positive training area; all evaluated cohort areas are positive and finite. Sparse groups have no minimum-count threshold and may be unstable. No evaluation prices contribute to any median.''',
         implementation='''
def fit_candidate(parameters):
    return MedianPricePerSqm().fit(train, y_train)

def predict_frame(estimator, frame):
    return estimator.predict(frame)
''', grid="candidates = [{}]",
         extra='''
_, fallback_levels = model.predict_with_fallback(test)
display(pd.Series(fallback_levels).value_counts().rename("Test listings").to_frame())
extra["test_fallback_counts"] = {str(k): int(v) for k, v in pd.Series(fallback_levels).value_counts().items()}
''',
         discussion="This baseline assumes total price scales linearly with area within a location and property type. Large areas can produce very large predictions. Hierarchical fallback enables unseen-group predictions but discards progressively more location detail."),
    dict(key="ridge", filename="03_ridge_regression.ipynb", title="Ridge Regression",
         objective="Quantify the predictive value of an additive model of physical and location attributes.",
         method="A scikit-learn Pipeline and ColumnTransformer fit training medians with missingness indicators, numeric standardization, and sparse one-hot encoding. Categories occurring fewer than five times are pooled, with at most 64 encoded levels per field; unseen categories map to the pooled level when present, otherwise an all-zero vector. Regularization is selected by validation MAE in VND.",
         implementation='''
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

def fit_candidate(parameters):
    estimator = Pipeline([
        ("preprocess", structured_preprocessor(scale=True)),
        ("regressor", Ridge(**parameters, solver="lsqr", tol=1e-5)),
    ])
    return estimator.fit(train, log_y_train)

def predict_frame(estimator, frame):
    return inverse_log(estimator.predict(frame), y_train)
''', grid='candidates = [{"alpha": alpha} for alpha in [1.0, 10.0, 100.0]]',
         discussion="The log-price model is additive in encoded features and cannot directly represent complex spatial or physical interactions. Coefficients are regularized associations rather than causal effects."),
    dict(key="forest", filename="04_random_forest.ipynb", title="Random Forest",
         objective="Assess nonlinear interactions using a reproducible ensemble of randomized regression trees.",
         method="Training-median numeric imputation with missingness indicators and bounded sparse one-hot categorical encoding precede the forest in a Pipeline. Scaling is omitted. Two prespecified configurations vary tree count, depth, split and leaf size, and feature subsampling. Two worker threads limit memory pressure.",
         implementation='''
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def fit_candidate(parameters):
    estimator = Pipeline([
        ("preprocess", structured_preprocessor(scale=False)),
        ("regressor", RandomForestRegressor(**parameters, random_state=SEED, n_jobs=2)),
    ])
    return estimator.fit(train, log_y_train)

def predict_frame(estimator, frame):
    return inverse_log(estimator.predict(frame), y_train)
''', grid='''
candidates = [
    {"n_estimators": 80, "max_depth": 18, "min_samples_split": 6,
     "min_samples_leaf": 3, "max_features": .5},
    {"n_estimators": 120, "max_depth": 24, "min_samples_split": 10,
     "min_samples_leaf": 5, "max_features": 1.0},
]
''', importance='''
importance_plot("forest", model.named_steps["preprocess"].get_feature_names_out(),
                model.named_steps["regressor"].feature_importances_,
                "Random forest impurity importance")
''', discussion="Impurity importance describes the fitted log-price forest and can favor high-cardinality features. Sparse categorical expansion and finite tree depth restrict model capacity. The prespecified search is intentionally small."),
    dict(key="lightgbm", filename="05_lightgbm.ipynb", title="LightGBM",
         objective="Evaluate gradient-boosted trees with native categorical splits and validation-based early stopping.",
         method="Numeric missing values are handled natively. Categorical levels are learned from training predictors, applied unchanged to validation and test, and supplied as pandas categorical columns. Unseen levels become missing. Early stopping uses validation log-RMSE; selection between fitted candidates uses original-scale validation MAE. The two objectives are distinct and are reported explicitly.",
         implementation='''
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

X_train, category_levels = native_features(train)
X_validation, _ = native_features(validation, category_levels)

def fit_candidate(parameters):
    estimator = LGBMRegressor(**parameters, random_state=SEED, n_jobs=2,
                              deterministic=True, force_col_wise=True, verbosity=-1)
    estimator.fit(X_train, log_y_train,
                  categorical_feature=CATEGORICAL,
                  eval_set=[(X_validation, log_y_validation)], eval_metric="rmse",
                  callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
    return estimator

def predict_frame(estimator, frame):
    features, _ = native_features(frame, category_levels)
    return inverse_log(estimator.predict(features, num_iteration=estimator.best_iteration_), y_train)
''', grid='''
candidates = [
    {"n_estimators": 1500, "learning_rate": .04, "num_leaves": 31,
     "max_depth": -1, "min_child_samples": 40, "reg_alpha": .1, "reg_lambda": 1.0},
    {"n_estimators": 1500, "learning_rate": .04, "num_leaves": 63,
     "max_depth": 10, "min_child_samples": 60, "reg_alpha": 1.0, "reg_lambda": 5.0},
]
''', extra='extra["best_iteration"] = int(model.best_iteration_)',
         importance='''
importance_plot("lightgbm", model.feature_name_, model.booster_.feature_importance(importance_type="gain"),
                "LightGBM gain importance")
''', discussion="Native category splits can represent location interactions without a large one-hot matrix. Gain importance reflects fitted log-price loss reduction, not causality. The temporal holdout may contain distribution shifts and new categorical levels."),
    dict(key="catboost", filename="06_catboost.ipynb", title="CatBoost",
         objective="Assess gradient boosting with native categorical processing and reproducible early stopping.",
         method="The eight categorical columns are explicitly passed via cat_features after missing strings become a fixed token. Numeric missing values remain NaN and use CatBoost's native handling. CatBoost computes ordered categorical statistics internally from training labels; counter_calc_method='SkipTest' also excludes evaluation predictors from categorical frequency counts. Evaluation labels enter only early stopping. Candidate stopping uses validation log-RMSE, while model selection uses validation MAE in VND. CPU execution uses two threads. An initial 1,000-iteration search with depths 6 and 8 exceeded the runner's 1,800-second cell budget on this machine. The reported search therefore caps iterations at 400, depth at 6, categorical-combination complexity at 1, and numeric borders at 64. This resource adjustment retains the frozen cohort and evaluation protocol; no test score selects the revised settings.",
         implementation='''
from catboost import CatBoostRegressor

X_train = catboost_features(train)
X_validation = catboost_features(validation)

def fit_candidate(parameters):
    estimator = CatBoostRegressor(**parameters, random_seed=SEED, thread_count=2,
                                   cat_features=CATEGORICAL, verbose=False,
                                   allow_writing_files=False, task_type="CPU")
    estimator.fit(X_train, log_y_train, eval_set=(X_validation, log_y_validation),
                  early_stopping_rounds=60, use_best_model=True, verbose=False)
    return estimator

def predict_frame(estimator, frame):
    return inverse_log(estimator.predict(catboost_features(frame)), y_train)
''', grid='''
candidates = [
    {"iterations": 400, "learning_rate": .05, "depth": 4,
     "l2_leaf_reg": 3.0, "loss_function": "RMSE", "max_ctr_complexity": 1,
     "border_count": 64, "counter_calc_method": "SkipTest"},
    {"iterations": 400, "learning_rate": .05, "depth": 6,
     "l2_leaf_reg": 7.0, "loss_function": "RMSE", "max_ctr_complexity": 1,
     "border_count": 64, "counter_calc_method": "SkipTest"},
]
''', extra='extra["best_iteration"] = int(model.get_best_iteration())',
         importance='''
importance_plot("catboost", model.feature_names_, model.feature_importances_,
                "CatBoost prediction-value importance")
''', discussion="Native categorical processing avoids external target encoding. Internal ordered statistics do not remove the need for temporal and duplicate controls. Prediction-value importance is model-specific and cannot be compared numerically with LightGBM gain or forest impurity importance."),
    dict(key="tfidf", filename="07_tfidf_ridge.ipynb", title="TF-IDF + Ridge",
         objective="Measure text-only predictive signal after conservative removal of explicit monetary information.",
         method="The primary experiment uses only the redacted concatenation of listing title and description; no structured columns are appended. A Pipeline fits training TF-IDF followed by Ridge on log prices. The predefined candidate set contrasts unigrams and unigram–bigram features and varies vocabulary budget, document-frequency limits, and alpha. It is a four-point budgeted search, not an exhaustive factorial experiment. Empty or unseen-token documents map to zero vectors.",
         implementation='''
def fit_candidate(parameters):
    estimator = text_pipeline(**parameters)
    return estimator.fit(text_inputs(train), log_y_train)

def predict_frame(estimator, frame):
    return inverse_log(estimator.predict(text_inputs(frame)), y_train)
''', grid='''
candidates = [
    {"ngram_range": (1, 1), "max_features": 30000, "min_df": 3, "max_df": .95, "alpha": 10.0},
    {"ngram_range": (1, 1), "max_features": 50000, "min_df": 5, "max_df": 1.0, "alpha": 100.0},
    {"ngram_range": (1, 2), "max_features": 30000, "min_df": 3, "max_df": .95, "alpha": 10.0},
    {"ngram_range": (1, 2), "max_features": 50000, "min_df": 5, "max_df": 1.0, "alpha": 100.0},
]
''', extra='''
extra["vocabulary_size"] = len(model.named_steps["tfidf"].vocabulary_)
display(pd.Series({"Training empty text fraction": text_inputs(train).eq("").mean(),
                   "Test empty text fraction": text_inputs(test).eq("").mean(),
                   "Fitted vocabulary size": extra["vocabulary_size"]}).to_frame("Value"))
''', discussion="Text redaction reduces the risk of copying the target from the advertisement, at the cost of deleting quantities such as area and room counts. Whitespace tokenization after accent normalization is reproducible but is not a dedicated Vietnamese word-segmentation model. Residual price-related semantics may persist."),
]


def build(spec):
    key = spec["key"]
    method = spec["method"].split(" An initial 1,000-iteration search")[0]
    cells = [
        markdown(f"# {spec['title']}\n\nTraining and evaluation on preprocessed property listings. EDA and dataset cleaning belong to `notebooks/`.\n\n## 1. Setup"),
        code(SETUP),
        markdown("## 2. Load Preprocessed Data\n\nThe default input is `data/processed/priced_listings.parquet`, exported by the EDA notebook. If it is absent, the existing `experiment_cohort.parquet` is used and its path is displayed below. Set `DATA_PATH` to select a file explicitly. Existing split labels are reused; otherwise a shared 70/15/15 split is created. Every input row is retained. No source download, EDA, cleaning, sampling, or deduplication runs here."),
        code(INPUT),
        markdown("## 3. Model\n\n" + method + "\n\nOnly training observations fit imputation, encoding, scaling, or TF-IDF. Nonbaseline estimators learn log1p prices and reverse the transformation for evaluation; predictions are bounded by the training price range. Target-derived columns are excluded from the predictors."),
        code(spec["implementation"]),
        markdown("## 4. Hyperparameters\n\nSelect the candidate with the lowest validation MAE in VND. Test targets are not used for selection."),
        code(spec["grid"] + '\ndisplay(pd.DataFrame(candidates))'),
        markdown("## 5. Train"),
        code(r'''
model, selected_parameters, tuning_results = select_candidate(
    candidates, fit_candidate, y_validation,
    lambda estimator: predict_frame(estimator, validation))
display(tuning_results)
display(pd.Series(selected_parameters, dtype=object, name="Selected value").to_frame())
'''),
        markdown("## 6. Evaluate and Save Results\n\nMAE and RMSE use original VND prices; R-squared is unitless and MAPE is a percentage. Metrics include every evaluation row. The residual histogram displays the central 98% range without changing scoring. Results and row-aligned predictions are saved under `reports/experiments/`."),
        code(f'''
validation_prediction = predict_frame(model, validation)
test_prediction = predict_frame(model, test)
extra = {{}}
{textwrap.dedent(spec.get('extra', '')).strip()}
result, evaluation_table = record_result(
    "{key}", manifest, parts, validation_prediction, test_prediction,
    selected_parameters, tuning_results, extra=extra)
display(evaluation_table.style.format({{"MAE": "{{:,.0f}}", "RMSE": "{{:,.0f}}",
                                      "R2": "{{:.4f}}", "MAPE": "{{:,.2f}}"}}))
diagnostics("{key}", test.price, test_prediction)
'''),
    ]
    if "importance" in spec:
        cells.append(code(spec["importance"]))
    cells.append(code(r'''
comparison = comparison_table(manifest)
display(comparison.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}",
                                 "R2": "{:.4f}", "MAPE": "{:,.2f}"}))
'''))
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3.12 (Real Estate .venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.10"},
    }
    for index, cell in enumerate(notebook.cells):
        cell.id = f"{key}-{index:02d}"
    nbf.write(notebook, ROOT / "models" / spec["filename"])
    print(spec["filename"])


if __name__ == "__main__":
    for spec in SPECS:
        build(spec)
