# 261-Programming-for-AI-and-DS

## Metadata
* Dataset: https://huggingface.co/datasets/vduydong/vietnam-real-estates

## Getting started
```
from datasets import load_dataset
import pandas as pd

# Load full dataset
ds = load_dataset("tinixai/vietnam-real-estates")
df = ds["train"].to_pandas()

# --- Example 1: Listings in Hanoi with known price ---
hanoi = df[
    (df["province_name"] == "Hà Nội") &
    (df["price"] > 0)
]
print(hanoi[["name", "price", "area", "bedroom_count"]].head())

# --- Example 2: Average price per province ---
avg_price = (
    df[df["price"] > 0]
    .groupby("province_name")["price"]
    .mean()
    .sort_values(ascending=False)
)
print(avg_price.head(10))

# --- Example 3: Filter apartments in Hanoi ---
apartments_hanoi = df[
    (df["property_type_name"].str.contains("Căn hộ", na=False)) &
    (df["province_name"] == "Hà Nội")
]
print(f"{len(apartments_hanoi):,} apartment listings in Hanoi")
```