# Property asking-price experiment summary

Source: [vduydong/vietnam-real-estates](https://huggingface.co/datasets/vduydong/vietnam-real-estates),
revision `a9a66ffa985edcf76b4be59ae2c6f5b1db889c38`. All five Parquet files were verified against their
published LFS SHA-256 checksums. Original files are unchanged and excluded from Git.

## Data and experimental protocol

- Inspected source rows: 1,000,000.
- Invalid/nonpositive prices: 78,419; invalid/nonpositive areas: 1;
  invalid timestamps: 0; explicit rental-only titles: 7,294.
  Exclusion counts overlap and must not be added.
- Eligible rows: 915,657. A seed-42 uniform sample of 60,000
  eligible rows was drawn across all five shards before modelling, independent of price magnitude.
- Removed sampled identity-proxy duplicates: 481; final cohort: 59,519.
- Training/validation/test sizes: 41,663 / 8,928 / 8,928.
- Training end: 2025-11-04 13:39:57.561000+00:00; validation end: 2025-12-02 22:12:39.046000+00:00.
  Offset-free timestamps are assigned UTC for ordering; their true timezone is unknown.
- Split fingerprint: `15340296092993fac87fe7f2cd02c9635da706a0510a79167d2be7fd83474af9`.
- Every preprocessing fit uses training observations. Validation MAE in VND selects candidates;
  boosting early stopping monitors validation log-RMSE. Test labels do not select candidates.
- Nonbaseline estimators fit log1p prices, inverse-transform predictions, and bound extrapolation
  by the maximum training price. Baselines fit original-scale targets. Models are not refitted
  on validation data. All reported metrics use complete original-scale partitions.

## Observed test results

| Model | MAE (VND) | RMSE (VND) | R² | MAPE (%) |
|---|---:|---:|---:|---:|
| Dummy Regressor | 45,548,775,416 | 2,966,493,482,061 | -0.0001 | 259.20 |
| Median Price per sqm | 39,768,818,191 | 2,968,655,086,029 | -0.0016 | 44.85 |
| Ridge Regression | 42,681,233,989 | 2,971,055,734,676 | -0.0032 | 74.73 |
| Random Forest | 37,358,517,200 | 2,965,843,657,508 | 0.0003 | 36.77 |
| LightGBM | 36,477,344,778 | 2,964,995,648,314 | 0.0009 | 28.59 |
| CatBoost | 37,207,498,640 | 2,964,647,677,804 | 0.0011 | 35.56 |
| TF-IDF + Ridge | 39,616,175,623 | 2,966,358,669,831 | -0.0000 | 54.23 |

MAE and RMSE are in VND; R² is unitless; MAPE is a percentage. All targets are positive,
but small or incorrectly extracted prices can disproportionately affect MAPE. No tail observations
were trimmed for scoring. These descriptive estimates do not establish statistical significance.

The largest recorded test price is **280,280,000,000,000 VND**. That single listing
accounts for **99.67%–99.99%** of
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
