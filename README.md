# Vietnam Real Estate Price Prediction

Dataset: [tinixai/vietnam-real-estates](https://huggingface.co/datasets/tinixai/vietnam-real-estates)

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline (Run in order)

| Notebook | Input | Output |
|---|---|---|
| `notebooks/01_eda_data-quality.ipynb` | HuggingFace dataset | `data/processed/priced_listings.parquet` |
| `notebooks/02_eda_price-analysis.ipynb` | `priced_listings.parquet` | Figures |
| `notebooks/03_features_engineering.ipynb` | `priced_listings.parquet` | `data/processed/model_input.parquet` |
| `notebooks/04_model_baselines.ipynb` | `model_input.parquet` | `reports/metrics/04_baselines.csv` |
| `notebooks/05_model_tree-ensembles.ipynb` | `model_input.parquet` | `reports/metrics/05_tree_ensembles.csv`, `reports/predictions/05_tree_ensembles.parquet` |
| `notebooks/06_model_text-nlp.ipynb` | `model_input.parquet` | `reports/metrics/06_text_tfidf.csv`, `reports/predictions/06_text_tfidf.parquet` |
| `notebooks/07_model_deep_tabular.ipynb` | `model_input.parquet` | `reports/metrics/07_deep_tabular.csv`, `reports/predictions/07_deep_tabular.parquet` |
| `notebooks/08_model_bamibert.ipynb` | `priced_listings.parquet` | `reports/metrics/08_bamibert.csv`, `reports/predictions/08_bamibert.parquet` |
| `notebooks/09_model_ensemble.ipynb` | `reports/predictions/*.parquet` | `reports/metrics/09_ensemble.csv` |

## Results (Test Set)

| Model | MAE (bn VND) | RMSE (bn VND) | R² | MAPE (%) |
|---|---:|---:|---:|---:|
| Dummy (median) | 11.70 | 36.65 | -0.059 | 98.29 |
| Median price per m² | 7.23 | 42.90 | -0.451 | 47.35 |
| Ridge | 5.73 | 23.87 | 0.551 | 31.97 |
| Random Forest | 5.41 | 22.06 | 0.616 | 31.22 |
| CatBoost | 5.24 | 21.06 | 0.650 | 31.66 |
| Ridge: text + structured | 4.94 | 21.61 | 0.632 | 28.09 |
| MLP + embeddings | 4.62 | 19.26 | 0.707 | 26.83 |
| XGBoost | 4.48 | 18.40 | 0.733 | 26.42 |
| LightGBM | 4.46 | 18.19 | 0.739 | 26.24 |
| BamiBERT (text only) | 4.36 | 17.35 | 0.763 | 26.86 |
| LightGBM + text | 4.22 | 17.32 | 0.764 | 25.05 |
| Blend: structured | 4.24 | 17.36 | 0.762 | 25.90 |
| Blend: structured + TF-IDF | 4.02 | 17.02 | 0.772 | 24.31 |
| Blend: structured + BamiBERT | 3.70 | 16.28 | 0.791 | 21.94 |
| **Blend: all** | **3.66** | **16.17** | **0.794** | **21.70** |
