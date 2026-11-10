# Model training on preprocessed data

EDA and dataset preprocessing belong to the existing notebooks in `notebooks/`.
The seven notebooks in `models/` only load the resulting data, train models,
evaluate them, and save results. They do not read raw shards, download datasets,
filter rows, create samples, deduplicate listings, or regenerate processed data.

## Input

The default input is `data/processed/priced_listings.parquet`, the output declared
by `notebooks/01_eda_data-quality.ipynb`. If it is absent, the existing
`data/processed/experiment_cohort.parquet` cache is used. Every notebook displays
its actual input path and split sizes. The EDA export is currently absent locally;
the committed execution results use the existing processed cache, unchanged.

Set `DATA_PATH` in a notebook to select another preprocessed Parquet file.
To choose the same file for all notebooks and the summary command, use:

```powershell
$env:EXPERIMENT_DATA_PATH = "data/processed/priced_listings.parquet"
```

An explicitly selected missing file raises an error; it never triggers a download
or preparation step. No large dataset files are committed.

The input must have numeric `price`, the seven numeric physical attributes, and
the eight categorical attributes listed in `experiments/data.py`. Price and area
must already be finite and positive. Numeric missing values remain available for
train-fitted model imputation. Invalid targets are reported rather than silently
dropped. Extra EDA columns such as `price_per_m2`, flags, and duplicate statistics
are not used as predictors.

TF-IDF uses an existing prepared `text` column when supplied. Otherwise it combines
`name` and/or `description`, removing monetary clauses and digit-bearing tokens
as model-specific feature construction to reduce direct target copying. Vocabulary
and IDF weights are learned only from training observations.

## Shared split

- A supplied `split` column containing `train`, `validation`, and `test` is preserved.
- Without split labels, `published_at` supplies a chronological 70/15/15 split.
  ISO-8601 dates accept variable fractional seconds; offset-free dates are assigned
  UTC for ordering. The true source timezone is not inferred.
- Without timestamps, a seed-42 random 70/15/15 split is used.
- All input rows are retained. Existing `row_id` values are preserved; otherwise
  row positions identify predictions. If `group_id` is supplied, overlapping groups
  across splits cause an error. Entity cleaning and deduplication remain upstream.

## Run

Use the existing Python 3.12 environment. From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-experiments.txt
.\.venv\Scripts\python.exe -X utf8 scripts/execute_model_notebooks.py
.\.venv\Scripts\python.exe -X utf8 scripts/summarize_experiments.py
```

Each notebook also runs independently in Jupyter from the project root or `models/`.
No EDA notebook needs to run again when its processed output already exists.
`requirements-experiments-lock.txt` records the validated Windows environment.

The models are mean-price DummyRegressor, district/type median price per square
metre, structured Ridge, Random Forest, LightGBM, CatBoost, and text-only TF-IDF +
Ridge. Validation MAE selects hyperparameters. Boosting additionally uses early
stopping on validation log-RMSE. Models are not refitted on validation data.
Imputation, scaling, encoding, group medians, and TF-IDF fit training data only.
CatBoost categorical frequency statistics exclude validation predictors.

Nonbaseline models fit log1p prices, invert predictions, and bound them using the
maximum training price. Every model is scored in original VND units with MAE,
RMSE, R-squared, and percentage MAPE. Supplied extreme values are retained in scoring.

## Results and validation

`reports/experiments/summary.md` and `comparison.csv` contain the actual results.
Per-model JSON files record the selected settings, metrics, input checksum, and
split fingerprint. Predictions remain local under `reports/experiments/predictions/`.
Figures are embedded in the notebooks and exported locally as PDFs. The summary
audit verifies prediction alignment, target values, metrics, and completed outputs.

Run controlled tests with:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

`scripts/build_model_notebooks.py` regenerates notebook source and clears outputs;
it is a maintenance tool, not a prerequisite for training. Run all notebooks after
changing shared training code. Incompatible older result rows are excluded from
intermediate comparison tables; the final audit requires all seven models to match.
