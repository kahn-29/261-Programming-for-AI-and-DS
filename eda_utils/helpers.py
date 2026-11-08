"""Helper functions for the Vietnam real-estate EDA notebook."""

import unicodedata

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------- reporting ----------
def title_with_n(base: str, frame_name: str, n: int) -> str:
    """Figure/table title that states source frame and row count."""
    return f"{base} [{frame_name}, n={n:,}]"


# ---------- duplicates ----------
def normalize_text(series: pd.Series) -> pd.Series:
    """Lowercase, strip accents, map Vietnamese đ->d, collapse whitespace."""
    s = series.fillna("").astype(str).str.lower()
    s = s.map(lambda x: unicodedata.normalize("NFKD", x))
    s = s.map(lambda x: "".join(ch for ch in x if not unicodedata.combining(ch)))
    s = s.str.replace("đ", "d", regex=False)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


# ---------- cleaning ----------
def prepare_priced_data(clean: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with positive price and area; add price_per_m2."""
    priced = clean.loc[clean["price"].gt(0) & clean["area"].gt(0)].copy()
    priced["price_per_m2"] = priced["price"] / priced["area"]
    return priced


# ---------- statistics ----------
def price_summary(frame, group_cols, min_n=100) -> pd.DataFrame:
    """Per-group listing count and median price / price_per_m2; groups with n >= min_n, sorted by n."""
    t = (
        frame.groupby(group_cols, observed=True)
        .agg(
            n=("price", "size"),
            median_price=("price", "median"),
            median_price_per_m2=("price_per_m2", "median"),
        )
        .reset_index()
    )
    return t[t["n"] >= min_n].sort_values("n", ascending=False).reset_index(drop=True)


def format_price_table(table: pd.DataFrame):
    """Styled copy of a price_summary table: price in billion VND, price per m2 in million VND."""
    out = table.copy()
    out["median_price (bn VND)"] = out.pop("median_price") / 1e9
    out["median_price_per_m2 (mil VND)"] = out.pop("median_price_per_m2") / 1e6
    fmt = {"n": "{:,}", "median_price (bn VND)": "{:.2f}", "median_price_per_m2 (mil VND)": "{:.1f}"}
    return out.style.format(fmt).hide(axis="index")


def pairwise_corr(frame, cols, method="pearson", min_n=1000):
    """Pairwise-deletion correlation. Returns (corr_matrix, n_matrix); pairs with n < min_n are NaN."""
    cols = list(cols)
    corr = frame[cols].corr(method=method, min_periods=min_n)
    ok = frame[cols].notna().astype(int)
    return corr, ok.T @ ok


# ---------- plotting ----------
def plot_box(frame, group_col, value_col, groups, ylabel, title, frame_name="df_priced"):
    """Boxplot (no outliers) of ``value_col`` by ``group_col`` for the given groups."""
    sub = frame.loc[frame[group_col].isin(groups), [group_col, value_col]]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=sub, x=group_col, y=value_col, order=groups, showfliers=False, ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title_with_n(title, frame_name, len(frame)))
    plt.tight_layout()
    plt.show()
