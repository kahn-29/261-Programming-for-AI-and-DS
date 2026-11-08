"""Regression metrics reported on the original VND scale.

All functions take `y_true` / `y_pred` in VND (NOT log). Convert with `10 ** pred_log` first.
"""

import numpy as np
import pandas as pd


def ape(y_true, y_pred) -> np.ndarray:
    """Absolute percentage error per row."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.abs(y_pred - y_true) / y_true


def regression_report(y_true, y_pred) -> dict:
    """Headline metrics: log-RMSE, MAE, MAPE, MdAPE, share within ±10% / ±20%, R² (log scale)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    e = ape(y_true, y_pred)
    log_err = np.log10(y_pred) - np.log10(y_true)
    ss_res = float(np.sum(log_err ** 2))
    ss_tot = float(np.sum((np.log10(y_true) - np.log10(y_true).mean()) ** 2))
    return {
        "n": int(len(y_true)),
        "rmse_log10": float(np.sqrt(np.mean(log_err ** 2))),
        "mae_vnd": float(np.mean(np.abs(y_pred - y_true))),
        "mape": float(np.mean(e)),
        "mdape": float(np.median(e)),
        "within_10pct": float(np.mean(e <= 0.10)),
        "within_20pct": float(np.mean(e <= 0.20)),
        "r2_log10": 1.0 - ss_res / ss_tot,
    }


def report_by_group(frame: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str, min_n: int = 100) -> pd.DataFrame:
    """regression_report per group (province, property type, ...). Groups below `min_n` are dropped."""
    rows = []
    for key, g in frame.groupby(group_col, observed=True):
        if len(g) >= min_n:
            rows.append({group_col: key, **regression_report(g[y_true_col], g[y_pred_col])})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
