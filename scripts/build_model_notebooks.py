"""Generate the seven research notebooks from a consistent narrative structure."""

from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def markdown(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


SETUP = '''
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
from IPython.display import display, Markdown
from experiments.data import load_data, SEED, NUMERIC, CATEGORICAL
from experiments.evaluation import (
    select_candidate, inverse_log, record_result, diagnostics,
    importance_plot, comparison_table, environment,
)
from experiments.estimators import (
    structured_preprocessor, native_features, catboost_features,
    text_pipeline, MedianPricePerSqm,
)

np.random.seed(SEED)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
parts, manifest = load_data()
train, validation, test = (parts[name] for name in ["train", "validation", "test"])
y_train = train.price.to_numpy()
y_validation = validation.price.to_numpy()
log_y_train = np.log1p(y_train)
log_y_validation = np.log1p(y_validation)
display(pd.Series({"Seed": SEED, "Dataset": manifest["source"]["dataset"],
                   "Revision": manifest["source"]["revision"],
                   "Split SHA-256": manifest["split_sha256"]}, name="Setting").to_frame())
display(pd.Series(environment()["packages"], name="Installed version").to_frame())
'''

OVERVIEW = '''
display(pd.DataFrame([
    {"Split": name, "Listings": len(frame),
     "First publication (UTC)": frame.published_at.min(),
     "Last publication (UTC)": frame.published_at.max()}
    for name, frame in parts.items()
]).set_index("Split"))
display(pd.DataFrame({"Training dtype": train[NUMERIC + CATEGORICAL].dtypes.astype(str),
                      "Training missing fraction": train[NUMERIC + CATEGORICAL].replace(
                          "__MISSING__", np.nan).isna().mean()}))
display(train.price.describe(percentiles=[.01, .5, .99]).rename("Training asking price (VND)").to_frame())
display(Markdown(
    f"All {manifest['counts']['raw_rows']:,} source rows were inspected; "
    f"{manifest['counts']['eligible_rows']:,} were eligible. A fixed-seed uniform sample of "
    f"{manifest['sampled_rows_before_deduplication']:,} eligible rows was drawn across all five shards. "
    f"Removing {manifest['sample_duplicates_removed']:,} repeated identity proxies left "
    f"{manifest['cohort_rows']:,} listings. These are sample-based experiments, not full-corpus model fits."
))
'''

PREPROCESSING = '''
Raw Parquet files remain unchanged. Eligibility requires finite, positive `price` and `area`,
and a parseable `published_at`. Titles explicitly advertising rental without sale or transfer
language are excluded using a fixed rule. The source has no transaction-type or verified-sale
label; undetected rentals and incorrectly extracted amounts remain possible. Other negative or
non-finite numeric attributes become missing. No price quantile filtering, target-dependent
outlier removal, or retrospective unit correction is applied.

ISO-8601 timestamps are parsed with and without fractional seconds. Offset-free timestamps
are assigned UTC for consistent ordering; the source does not establish their true timezone.
Publication timestamps cover multiple months. A chronological 70/15/15 split approximates
prediction on subsequently published listings. Boundaries are fixed once from feature timestamps.
Within the sampled cohort, an identity proxy combines a normalized title with numeric substrings
masked, province, district, ward, and area rounded to two decimals. Only its earliest sampled
record is retained before splitting. This target-free proxy is unique across all partitions;
it is not a verified property identifier and cannot eliminate every repost or near duplicate.
There is no spatial holdout, and performance outside the observed locations is unmeasured.

Only training data fit imputation, scaling, encodings, category vocabularies, group medians,
or TF-IDF. Validation selects candidate hyperparameters and early-stopping iterations. The
selected training-fitted model is evaluated once on the test partition; validation is not
merged back into training. Test outcomes are descriptive and must not be used to revise the search.
'''

FEATURES = '''
Structured models use the seven physical attributes (`area`, `floor_count`, `frontage_width`,
`house_depth`, `road_width`, `bedroom_count`, `bathroom_count`) and eight categorical attributes
(`property_type_name`, `province_name`, `district_name`, `ward_name`, `street_name`,
`project_name`, `house_direction`, `balcony_direction`). Price, price per square metre,
duplicate keys, source row identifiers, and split labels are excluded from model inputs.
Publication time controls the split and is not a predictor.

The text-only experiment concatenates `name` and `description`. Fixed normalization removes
Vietnamese accents; monetary clauses and all tokens containing digits are discarded, including
embedded numeric price abbreviations. This conservative redaction limits direct price copying
but also removes useful physical quantities. It cannot establish that all semantic price cues
are absent. No test-dependent redaction or vocabulary fitting is performed.

Nonbaseline estimators fit `log1p(price)` to reduce sensitivity during training to the long
price tail. Inverse-transformed predictions are bounded between zero and the maximum training
price using training information only. This bound stabilizes extrapolation and can underpredict
new high-price listings. No lognormal bias correction is applied. All selection and reported
metrics use original VND prices. The two baselines fit original-scale prices directly.
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
    return estimator.fit(train.text, log_y_train)

def predict_frame(estimator, frame):
    return inverse_log(estimator.predict(frame.text), y_train)
''', grid='''
candidates = [
    {"ngram_range": (1, 1), "max_features": 30000, "min_df": 3, "max_df": .95, "alpha": 10.0},
    {"ngram_range": (1, 1), "max_features": 50000, "min_df": 5, "max_df": 1.0, "alpha": 100.0},
    {"ngram_range": (1, 2), "max_features": 30000, "min_df": 3, "max_df": .95, "alpha": 10.0},
    {"ngram_range": (1, 2), "max_features": 50000, "min_df": 5, "max_df": 1.0, "alpha": 100.0},
]
''', extra='''
extra["vocabulary_size"] = len(model.named_steps["tfidf"].vocabulary_)
display(pd.Series({"Training empty text fraction": train.text.eq("").mean(),
                   "Test empty text fraction": test.text.eq("").mean(),
                   "Fitted vocabulary size": extra["vocabulary_size"]}).to_frame("Value"))
''', discussion="Text redaction reduces the risk of copying the target from the advertisement, at the cost of deleting quantities such as area and room counts. Whitespace tokenization after accent normalization is reproducible but is not a dedicated Vietnamese word-segmentation model. Residual price-related semantics may persist."),
]


def build(spec):
    key = spec["key"]
    cells = [
        markdown(f"# {spec['title']}: Property Asking-Price Prediction\n\n## 1. Title and Objective\n\n{spec['objective']}\n\nSource: [vduydong/vietnam-real-estates](https://huggingface.co/datasets/vduydong/vietnam-real-estates), revision `a9a66ffa985edcf76b4be59ae2c6f5b1db889c38`. The dataset credits TiniX AI and specifies CC BY-NC 4.0. The outcome is the recorded asking price in VND, not a verified transaction price."),
        markdown("## 2. Experimental Setup\n\nRun `python -m experiments.prepare` once from the repository root. This notebook then executes independently from the repository root or `models/`, uses the frozen cohort, and never downloads data implicitly. Exact dependencies are recorded in `requirements-experiments-lock.txt`."),
        code(SETUP), markdown("## 3. Dataset Overview\n\nThe following tables are computed from the verified cohort. Feature missingness and price summaries describe training data only."),
        code(OVERVIEW), markdown("## 4. Data Preprocessing\n\n" + PREPROCESSING),
        code('''
assert set(NUMERIC + CATEGORICAL).isdisjoint({"price", "price_per_m2", "group_id", "row_id", "split"})
assert train.published_at.max() < validation.published_at.min()
assert validation.published_at.max() < test.published_at.min()
assert train.group_id.is_unique and validation.group_id.is_unique and test.group_id.is_unique
assert set(train.group_id).isdisjoint(validation.group_id)
assert set(train.group_id).isdisjoint(test.group_id)
assert set(validation.group_id).isdisjoint(test.group_id)
'''),
        markdown("## 5. Feature Engineering\n\n" + FEATURES),
        markdown("## 6. Model Implementation\n\n" + spec["method"]), code(spec["implementation"]),
        markdown("## 7. Hyperparameter Configuration\n\nThe following candidates are prespecified. Selection minimizes validation MAE in VND. Baselines have one fixed configuration; they do not use validation targets to fit their parameters."),
        code(spec["grid"] + '\ndisplay(pd.DataFrame(candidates))'),
        markdown("## 8. Model Training\n\nEach candidate fits training observations. The selected fitted candidate is retained without refitting on validation data. Reported times include fitting and validation prediction and depend on hardware."),
        code('''
model, selected_parameters, tuning_results = select_candidate(
    candidates, fit_candidate, y_validation,
    lambda estimator: predict_frame(estimator, validation))
display(tuning_results)
display(pd.Series(selected_parameters, dtype=object, name="Selected value").to_frame())
'''),
        markdown("## 9. Evaluation\n\nMAE and RMSE are expressed in VND. R² is unitless; MAPE is a percentage and is defined because all eligible prices are strictly positive. Very small recorded prices can dominate MAPE, so it is secondary to MAE. Metrics use the complete partitions, including extreme values. The residual histogram displays only the central 98% range; no observations are removed from scoring. Scatterplots display at most 3,000 fixed-seed test observations."),
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
    cells.extend([
        markdown("## 10. Results and Discussion\n\n" + spec["discussion"] + "\n\nThe comparison table contains only completed models with an identical protocol and split fingerprint. Missing or incompatible previous model results have no row until re-executed; no scores are imputed. The final summary audit requires all seven models to match. Rankings are descriptive for this cohort. They do not establish statistical significance or performance on verified sales."),
        code('''
comparison = comparison_table(manifest)
display(comparison.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}",
                                 "R2": "{:.4f}", "MAPE": "{:,.2f}"}))
score = result["metrics"]["test"]
display(Markdown(
    f"The held-out test evaluation gives MAE **{score['MAE']:,.0f} VND**, "
    f"RMSE **{score['RMSE']:,.0f} VND**, R² **{score['R2']:.4f}**, "
    f"and MAPE **{score['MAPE']:,.2f}%**. These observed values describe "
    "the frozen temporal cohort and the recorded asking-price labels."
))
'''),
        markdown("## 11. Conclusion\n\nThis notebook provides an executed, auditable estimate under the shared temporal protocol. Source checksums, split identity, candidate settings, environment versions, predictions, and original-scale metrics are recorded as artifacts. Interpretation is limited by sample-based training, a heuristic entity proxy, extreme or incorrectly extracted prices, incomplete attributes, possible mixed transaction types, and a single temporal holdout. A separate verified-sale dataset and repeated temporal backtesting are required before deployment or broad market claims."),
    ])
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3.12 (Real Estate .venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.10"},
    }
    for index, cell in enumerate(notebook.cells):
        cell.id = f"{key}-{index:02d}"
    path = ROOT / "models" / spec["filename"]
    nbf.write(notebook, path)
    print(path.name)


if __name__ == "__main__":
    for spec in SPECS:
        build(spec)
