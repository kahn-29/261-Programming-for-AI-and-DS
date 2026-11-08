# Vietnam Real Estate: EDA and Price Modelling

Dataset: https://huggingface.co/datasets/tinixai/vietnam-real-estates

## Run

    pip install -r requirements.txt
    # optional, for notebooks 06-07:
    pip install -r requirements-extra.txt

Run the notebooks in `notebooks/` in numeric order. Notebook 01 downloads the data into `dataset/` on first run and writes the cleaned tables to `data/processed/`.

## Layout

    dataset/            raw download (git-ignored)
    data/interim/       intermediate files (git-ignored)
    data/processed/     tables used by the notebooks (git-ignored)
    notebooks/          numbered pipeline, see below; sandbox/ for personal experiments
    eda_utils/          helpers and constants for the EDA notebooks
    ml_utils/           config, splits, metrics, features, model factories for modelling
    models/             saved models (git-ignored)
    reports/figures/    exported figures

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_eda_data-quality` | schema, missing values, duplicates, builds `df_clean` and `df_priced` |
| `02_eda_price-analysis` | price by location and type, correlations, anomalies |
| `03_features_engineering` | target, train/test split, features |
| `04_model_baselines` | naive group medians, linear models |
| `05_model_tree-ensembles` | Random Forest, LightGBM, XGBoost, CatBoost |
| `06_model_text-nlp` | TF-IDF, sentence embeddings, PhoBERT |
| `07_model_deep-tabular` | TabPFN, MLP, FT-Transformer, TabM |
| `08_model_ensemble` | blending and stacking |
| `09_eval_error-analysis-shap` | final test scores, error analysis, SHAP |

## Conventions

- Notebooks: `NN_stage_topic.ipynb`, lowercase, no spaces or accents, `_` between parts and `-` inside a part. One notebook answers one question. Do not use `final`, `v2`, `new`.
- Every notebook starts with goal, inputs, outputs, previous/next, seed.
- Data: `<stage>_<what>.parquet`. Models: `<algorithm>_<target>_v<N>.<ext>`. Figures: `fig<NN>_<topic>.png`.
- Code: reusable logic lives in `eda_utils/` or `ml_utils/`, notebooks only orchestrate and show results.
- Splits: final test = latest listings by `published_at`; cross-validation uses `GroupKFold` on `soft_key` so duplicates do not leak. Never tune on the test set.
- Commits: `<type>: <message>` (`feat`, `fix`, `docs`, `chore`, `exp`). Experiments on branches `exp/<name>`.

## Known pitfalls

- `price` and `price_per_m2` are the same information; never use one as a feature for the other.
- `description` often contains the price; strip money amounts before using text.
- Only listings with a disclosed price are modelled, so conclusions apply to those.
