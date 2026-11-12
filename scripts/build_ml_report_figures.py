"""Build report figures from the saved outputs of notebooks 03--06."""

import ast
import base64
import json
import re
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


def notebook(name):
    return json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))


def output_text(nb, cell):
    outputs = nb["cells"][cell].get("outputs", [])
    return "\n".join(
        "".join(o.get("text", []))
        or "".join(o.get("data", {}).get("text/plain", []))
        for o in outputs
    )


def extract_png(nb, cell, name):
    outputs = nb["cells"][cell].get("outputs", [])
    figures = [o["data"]["image/png"] for o in outputs if "image/png" in o.get("data", {})]
    if len(figures) != 1:
        raise ValueError(f"Expected one saved figure in cell {cell}")
    payload = figures[0]
    if isinstance(payload, list):
        payload = "".join(payload)
    (OUT / name).write_bytes(base64.b64decode(payload))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


baselines = notebook("04_model_baselines.ipynb")
trees = notebook("05_model_tree-ensembles.ipynb")
text = notebook("06_model_text-nlp.ipynb")
extract_png(baselines, 15, "ml_ridge_actual_vs_predicted.png")
extract_png(trees, 12, "ml_lightgbm_feature_importance.png")

# Read validation scores from notebook outputs; do not substitute test scores.
ridge_match = re.search(r"Validation MAE \(bn VND\): (\{[^\n]+\})", output_text(baselines, 11))
ridge = ast.literal_eval(ridge_match.group(1))
text_rows = re.findall(r"^\s*\d+\s+(1|10|100)\s+(\d+\.\d+)\s*$", output_text(text, 10), re.M)
text_ridge = {int(alpha): float(score) for alpha, score in text_rows}
forest_rows = re.findall(r"^\s*\d+\s+(12|20)\s+(0\.5|1\.0)\s+(\d+\.\d+)\s*$", output_text(trees, 8), re.M)
assert len(ridge) == len(text_ridge) == 3 and len(forest_rows) == 4

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained")
alphas = [1, 10, 100]
for values, label, color in [(ridge, "Structured Ridge", "#28689b"),
                              (text_ridge, "Text + structured Ridge", "#d87831")]:
    ys = [values[a] for a in alphas]
    axes[0].plot(range(3), ys, "o-", color=color, label=label)
    for x, y in enumerate(ys):
        axes[0].annotate(f"{y:.3f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=11)
axes[0].set(xticks=range(3), xticklabels=alphas, xlabel=r"Regularisation $\alpha$",
            ylabel="Validation MAE (billion VND)", title="(a) Ridge regularisation search", ylim=(4.35, 5.65))
axes[0].legend(loc="center left", fontsize=10)
labels = [f"depth {depth}\nfeatures {fraction}" for depth, fraction, _ in forest_rows]
scores = [float(score) for _, _, score in forest_rows]
best = int(np.argmin(scores))
bars = axes[1].bar(range(4), scores, color=["#28689b" if i == best else "#a7bbc9" for i in range(4)], width=.65)
axes[1].bar_label(bars, fmt="%.3f", padding=4, fontsize=11)
axes[1].set(xticks=range(4), xticklabels=labels, xlabel="Random Forest configuration",
            ylabel="Validation MAE (billion VND)", title="(b) Random Forest grid search", ylim=(0, 6.9))
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
save(fig, "ml_validation_selection")

fig, ax = plt.subplots(figsize=(11, 4.4))
ax.set(xlim=(0, 11), ylim=(-.1, 4.4))
ax.axis("off")


def box(x, y, w, h, label, color="#eaf1f6", size=11):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07", linewidth=.9,
                          edgecolor="#436176", facecolor=color)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=size)


def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                linewidth=1.1, color="#436176"))


box(.15, 3.55, 10.7, .6, "Shared modelling table  |  hierarchical locations + missing categories + cleaned text\nChronological split: train 70% / validation 15% / test 15%", "#dcebdc", 12)
for x in [1.75, 5.5, 9.15]:
    arrow((x, 3.48), (x, 3.05))
box(.15, 2.35, 3.2, .65, "Baseline and linear inputs\nRidge: log1p + impute + scale\nRidge: one-hot categories")
box(3.85, 2.35, 3.3, .65, "Tree models\nordinal categories\nunscaled numeric features")
box(7.7, 2.35, 3.15, .65, "Text models\ntraining-fitted TF-IDF", "#fff0dd")
arrow((1.75, 2.28), (1.75, 1.9))
arrow((5.5, 2.28), (5.5, 1.9))
arrow((9.15, 2.28), (9.15, 1.9))
box(.15, 1.15, 3.2, .7, "Median baselines\nStructured Ridge: tune alpha")
box(3.85, 1.15, 3.3, .7, "Random Forest / LightGBM\nCatBoost / XGBoost")
box(7.7, 1.15, 3.15, .7, "TF-IDF + structured -> Ridge\nOOF text -> LightGBM\n(+ structured inputs)", "#fff0dd")
for x in [1.75, 5.5, 9.15]:
    arrow((x, 1.08), (x, .75))
box(.15, .05, 10.7, .65, "Learned regressors: log1p(price) -> expm1 prediction -> test metrics in VND\nPreprocessing fits training rows; final refits differ by model.", "#e9e5f2", 12)
save(fig, "ml_training_pipeline")
print("Built four report figures from notebooks 03--06.")
