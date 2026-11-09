"""Bundle code, frozen cohort and manifest into one zip for Google Colab.

Run from the repository root:
    .venv/bin/python scripts/pack_colab.py
Upload dist/colab_bundle.zip to Google Drive, then follow EXPERIMENTS.md.
Raw source shards are not included; the cohort is verified by its SHA-256.
"""

from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [
    "experiments/*.py", "tests/*.py", "scripts/*.py", "models/*.ipynb",
    "requirements*.txt", "EXPERIMENTS.md",
    "reports/experiments/*.json", "reports/experiments/*.md", "reports/experiments/*.csv",
    "data/processed/experiment_cohort.parquet",
    "data/processed/text_vi.parquet"
]


def main():
    cohort = ROOT / "data/processed/experiment_cohort.parquet"
    if not cohort.exists():
        raise FileNotFoundError("Run python -m experiments.prepare first.")
    output = ROOT / "dist/colab_bundle.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        for pattern in INCLUDE:
            for path in sorted(ROOT.glob(pattern)):
                bundle.write(path, path.relative_to(ROOT).as_posix())
    print(f"Wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
