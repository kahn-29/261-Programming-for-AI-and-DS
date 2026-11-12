"""Render neural-model and conclusion figures using recorded experiment outputs."""

import base64
import csv
import json
import re
from html import unescape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "latex" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})


def read_notebook(name):
    return json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))


def save(fig, name):
    fig.savefig(OUT / name, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


mlp = read_notebook("07_model_deep_tabular")
bami = read_notebook("08_model_bamibert")
blend = read_notebook("09_model_ensemble")
cardinality_text = "\n".join("".join(o.get("data", {}).get("text/plain", [])) for o in mlp["cells"][6]["outputs"])
cardinalities = [int(n) for n in re.findall(r"^\w+\s+(\d+)\s*$", cardinality_text, re.M)]
assert len(cardinalities) == 8
embedding_components = sum(min(32, round(1.6 * n ** .56)) for n in cardinalities)
model_text = "\n".join("".join(o.get("data", {}).get("text/plain", [])) for o in mlp["cells"][8]["outputs"])
combined_inputs = int(re.search(r"Linear\(in_features=(\d+)", model_text).group(1))
numeric_components = combined_inputs - embedding_components

# Preserve the actual text-model learning curve; no synthetic intermediate points.
images = [o["data"]["image/png"] for o in bami["cells"][13]["outputs"] if "image/png" in o.get("data", {})]
assert len(images) == 1
payload = images[0]
(OUT / "dl_bamibert_training.png").write_bytes(base64.b64decode("".join(payload) if isinstance(payload, list) else payload))

# MLP history is printed to four decimals; the figure retains that precision.
logs = "".join("".join(o.get("text", [])) for o in mlp["cells"][11]["outputs"])
rows = re.findall(r"epoch\s+(\d+)\s+train loss\s+([\d.]+)\s+val MAE \(log\)\s+([\d.]+)", logs)
history = np.array(rows, dtype=float)
assert history.shape == (13, 3)
best = int(np.argmin(history[:, 2]))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
for ax, column, label, color in [(axes[0], 1, "Training Huber loss", "#28689b"),
                                  (axes[1], 2, "Validation MAE (log price)", "#d87831")]:
    ax.plot(history[:, 0], history[:, column], "o-", color=color, markersize=4)
    ax.axvline(history[best, 0], color="#555555", linestyle="--", linewidth=1)
    ax.set(xlabel="Epoch", ylabel=label, xticks=[1, 4, 7, 10, 13])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.2)
axes[0].set_title("(a) Optimisation history")
axes[1].set_title("(b) Validation and checkpoint selection")
axes[1].annotate("Restored epoch 10\nMAE = 0.2246", (10, history[best, 2]), xytext=(5.7, .244),
                 arrowprops={"arrowstyle": "->", "color": "#555555"}, fontsize=11)
save(fig, "dl_mlp_training.png")

# Two architectures, reflecting the implemented layers rather than a generic model.
fig, ax = plt.subplots(figsize=(10.8, 6.1))
ax.set(xlim=(0, 11), ylim=(0, 8.2))
ax.axis("off")


def box(x, y, w, h, label, color="#eaf1f6", size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.06", linewidth=1,
                               edgecolor="#436176", facecolor=color))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=size)


def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                color="#436176", linewidth=1.1))


ax.text(2.6, 7.85, "(a) Structured MLP", ha="center", weight="bold", fontsize=13)
ax.text(8.4, 7.85, "(b) Text-only BamiBERT regressor", ha="center", weight="bold", fontsize=13)
box(.15, 6.3, 2.3, 1.05, "7 numeric features\nlog1p / impute\nmissing flags / scale")
box(2.75, 6.3, 2.3, 1.05, "8 categorical features\nrare-level pooling\nlearned embeddings")
arrow((1.3, 6.24), (2.6, 5.75))
arrow((3.9, 6.24), (2.6, 5.75))
box(.15, 4.8, 4.9, .85, f"Concatenate: {combined_inputs} inputs\n{numeric_components} numeric + {embedding_components} embedding components")
arrow((2.6, 4.74), (2.6, 4.24))
box(.15, 3.2, 4.9, .95, "Dense 256 → ReLU → Dropout 0.2\nDense 256 → ReLU → Dropout 0.2")
arrow((2.6, 3.14), (2.6, 2.64))
box(.15, 1.65, 4.9, .9, "Linear output: predicted log1p(price)\nAdamW + Huber loss", "#dcebdc")
arrow((2.6, 1.59), (2.6, 1.06))
box(.15, .15, 4.9, .8, "expm1 → asking price in VND", "#e9e5f2")

box(5.95, 6.3, 4.9, 1.05, "Title + description\nmonetary clauses removed\nother numeric content retained", "#fff0dd")
arrow((8.4, 6.24), (8.4, 5.75))
box(5.95, 4.8, 4.9, .85, "First 3,000 characters → tokenizer\n512-token maximum; dynamic batch padding", "#fff0dd")
arrow((8.4, 4.74), (8.4, 4.24))
box(5.95, 3.2, 4.9, .95, "Fine-tuned BamiBERT encoder\n+ sequence regression head", "#fff0dd")
arrow((8.4, 3.14), (8.4, 2.64))
box(5.95, 1.65, 4.9, .9, "One output: standardised log price\nMSE loss; one training epoch", "#dcebdc")
arrow((8.4, 1.59), (8.4, 1.06))
box(5.95, .15, 4.9, .8, "Undo target scaling → expm1 → VND", "#e9e5f2")
save(fig, "dl_model_architectures.png")

# Coefficients are available as a formatted table, hence two-decimal precision.
html = "".join("".join(o.get("data", {}).get("text/html", [])) for o in blend["cells"][9]["outputs"])
headers = [unescape(re.sub("<[^>]+>", "", h)).strip() for h in re.findall(r'<th[^>]*class="col_heading[^\"]*"[^>]*>(.*?)</th>', html, re.S)]
body = re.search(r"<tbody>(.*?)</tbody>", html, re.S).group(1)
names, weights = [], []
for row in re.findall(r"<tr>(.*?)</tr>", body, re.S):
    names.append(re.sub("<[^>]+>", "", re.search(r"<th[^>]*>(.*?)</th>", row, re.S).group(1)).strip().replace("Blend: ", ""))
    cells = [re.sub("<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
    weights.append([float(c) if c else np.nan for c in cells])
matrix = np.array(weights)
assert matrix.shape == (4, 7) and len(headers) == 7
fig, ax = plt.subplots(figsize=(10, 3.1), layout="constrained")
cmap = plt.get_cmap("Blues").copy()
cmap.set_bad("#eeeeee")
img = ax.imshow(matrix, vmin=0, vmax=.5, cmap=cmap, aspect="auto")
labels = {"Ridge text + structured": "Ridge\ntext + structured", "LightGBM + text": "LightGBM\n+ text"}
ax.set(xticks=range(7), xticklabels=[labels.get(h, h) for h in headers],
       yticks=range(4), yticklabels=names)
ax.tick_params(axis="both", length=0, pad=8)
for i in range(4):
    for j in range(7):
        v = matrix[i, j]
        ax.text(j, i, "—" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                color="white" if v > .28 else "#222222", fontsize=10)
fig.colorbar(img, ax=ax, label="Nonnegative coefficient", fraction=.04, pad=.025)
save(fig, "dl_ensemble_coefficients.png")

with (ROOT / "reports/metrics/09_ensemble.csv").open(encoding="utf-8") as f:
    metrics = {r[""]: r for r in csv.DictReader(f)}
order = ["LightGBM", "MLP", "BamiBERT", "LightGBM + text", "Blend: structured",
         "Blend: structured + TF-IDF", "Blend: structured + BamiBERT", "Blend: all"]
names = ["LightGBM", "MLP + embeddings", "BamiBERT (text only)", "LightGBM + text",
         "Structured blend", "Structured + TF-IDF blend", "Structured + BamiBERT blend", "All-model blend"]
scores = [float(metrics[k]["MAE (bn VND)"]) for k in order]
fig, ax = plt.subplots(figsize=(9.4, 4.2), layout="constrained")
bars = ax.barh(range(len(order)), scores, color=["#7597b0"]*4+["#b8a5c9"]*3+["#287e68"], height=.65)
ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=10)
ax.set(yticks=range(len(names)), yticklabels=names, xlim=(0, 5.15),
       xlabel="Test MAE (billion VND); lower is better")
ax.invert_yaxis()
ax.spines[["top", "right", "left"]].set_visible(False)
ax.grid(axis="x", alpha=.2)
ax.set_axisbelow(True)
save(fig, "conclusion_model_comparison.png")
print("Built five figures: architectures, two training diagnostics, blend coefficients, and test MAE comparison.")
