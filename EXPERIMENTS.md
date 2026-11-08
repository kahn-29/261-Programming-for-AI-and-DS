# Reproducible property asking-price experiments

The seven notebooks in `models/` implement a shared temporal benchmark using
[vduydong/vietnam-real-estates](https://huggingface.co/datasets/vduydong/vietnam-real-estates),
revision `a9a66ffa985edcf76b4be59ae2c6f5b1db889c38`. The source card credits
TiniX AI and specifies CC BY-NC 4.0. This benchmark uses the explicitly selected
one-million-row source; the earlier EDA notebook retains its original data source.

## Running the experiments

Use Python 3.12. From the project root on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-experiments-lock.txt
.\.venv\Scripts\python.exe -X utf8 -m experiments.prepare
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 scripts/execute_model_notebooks.py
.\.venv\Scripts\python.exe -X utf8 scripts/summarize_experiments.py
```

`requirements-experiments.txt` lists direct requirements on top of the existing
project requirements. `requirements-experiments-lock.txt` captures the exact
installed package versions used for the executed results, including the existing
project dependencies; its Windows-specific packages target Windows reproduction.

Preparation downloads approximately 469 MB across five source shards, verifies
their published LFS checksums, profiles every source row in bounded-memory batches,
and writes an immutable-source manifest and a derived modelling cohort. It never
overwrites raw Parquet files. An existing source file with an unexpected checksum
causes an error. There are no implicit downloads when opening a model notebook.

After preparation, any notebook executes independently from the project root or
`models/`; select the project's `.venv` kernel. Sequential execution limits memory
pressure. Both the model search and numeric-library thread counts are bounded.
Runtime depends on the machine; actual runtimes are in the execution report.

The initial CatBoost grid (1,000 iterations; depths 6 and 8) exceeded the
1,800-second cell execution budget. The final, fully executed experiment uses
400 iterations, depths 4 and 6, at most one categorical feature per internal
statistic, and 64 numeric borders. The data and evaluation split were unchanged;
the adjustment addressed computation rather than observed test performance.

To execute from the notebook directory through the same runner:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/execute_model_notebooks.py --from-models
```

The generator `scripts/build_model_notebooks.py` is provided for maintenance.
Running it replaces notebook contents and clears outputs; it is not part of normal
reproduction. Re-execute the notebooks after any regeneration. Editing shared
experimental source invalidates the recorded protocol identity; rerun all seven
models and the summary after such changes.

## Models

| Notebook | Estimator and representation | Validation search |
|---|---|---|
| `01_baseline_dummy_regressor.ipynb` | Mean-price DummyRegressor | Fixed strategy |
| `02_baseline_median_price.ipynb` | Training district/type median price per square metre multiplied by area | Fixed hierarchical fallback |
| `03_ridge_regression.ipynb` | Median imputation, scaling, sparse one-hot encoding, Ridge | Three alpha values |
| `04_random_forest.ipynb` | Median imputation, sparse one-hot encoding, RandomForestRegressor | Two tree configurations |
| `05_lightgbm.ipynb` | Native categorical and numeric missing-value handling | Two configurations, early stopping |
| `06_catboost.ipynb` | Explicit native categorical features and native numeric missing-value handling | Two configurations, early stopping |
| `07_tfidf_ridge.ipynb` | Redacted title/description, TF-IDF, Ridge | Four unigram/bigram configurations |

Median-baseline keys qualify district names by province. Unseen district/type
combinations fall back to province/type, type, then global training unit-price
medians. Invalid inference areas use the positive training-area median. Other
models exclude price-per-area and every other target-derived feature.

## Protocol and scope

The full source is inspected, then a fixed-seed uniform sample of 60,000 eligible
rows is drawn across all shards. A target-free title/location/area proxy removes
sampled repeated listings, retaining the earliest sampled observation. Distinct
proxy groups are divided chronologically into approximately 70% training, 15%
validation, and 15% test. ISO-8601 timestamps accept variable fractional seconds;
offset-free timestamps are assigned UTC for ordering only. No verified entity ID
or true source timezone is available.

Eligibility requires finite positive prices and areas and valid publication times.
Explicit rental-only titles are excluded by a fixed rule. Other invalid numerical
features become missing. No target quantile trimming or price-unit correction is
performed. All five observed property types are retained; this is property
asking-price prediction rather than a detached-house-only or verified-sale study.

Every learned transformation fits training data. Validation selects models using
MAE in VND. Early stopping for boosting separately monitors validation log-RMSE.
Models are not refitted on validation data. Nonbaseline estimators learn log1p
prices, then predictions are inverse-transformed and bounded by training support.
Baselines learn prices on their original scale. All metrics use original VND units.
MAPE is secondary because very small positive recorded prices can dominate it.

Text clauses containing monetary cues are discarded, followed by removal of every
digit-bearing token. This reduces direct target copying and also removes physical
quantities; semantic price cues may remain. No language model or external corpus is
downloaded. Tokenization is whitespace based after accent normalization.

The bounded cohort is a resource decision (approximately 3.4 GiB RAM was available
at preparation), not a claim that 60,000 rows saturate performance. Limitations
include noisy prices, imperfect entity matching, mixed transaction types, geographic
dependence, no uncertainty intervals, and a single temporal holdout. These must
accompany any academic interpretation. No model is declared superior without its
observed evaluation, and descriptive rankings do not imply significance.

## Artifacts

- `data/raw/vduydong/`: verified, unchanged raw source files; ignored by Git.
- `data/processed/experiment_cohort.parquet`: reproducible derived cohort; ignored.
- `reports/experiments/data_manifest.json`: source checksums, full-source null
  counts and distributions, cohort filtering, sample budget, temporal boundaries,
  and split/cohort fingerprints.
- `reports/experiments/{model}.json`: selected configuration, validation search,
  original-scale scores, package versions, protocol identity, and prediction hashes.
- `reports/experiments/predictions/`: local row-aligned validation/test predictions;
  excluded from Git, reproducible by running the notebooks.
- `reports/experiments/summary.md` and `comparison.csv`: consolidated real results.
- `reports/experiments/comparison.pdf`: standalone comparison figure.
- `reports/experiments/figures/`: local standalone diagnostic/importance PDFs;
  diagnostic PNGs are also embedded in the executed notebooks.
- `reports/experiments/validation_audit.json`: recomputed-metric, row-alignment,
  checksum, finite-output, split, and notebook-output checks.

The existing `main.ipynb`, EDA helpers, and unfinished `notebooks/` files are
preserved. Earlier environment-setup changes are kept separate from this task's
commit. Raw listing texts, model binaries, caches, and predictions are not committed.
