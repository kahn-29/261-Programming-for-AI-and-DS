"""Run with python -m experiments.prepare from the repository root."""

from .data import prepare_data

if __name__ == "__main__":
    _, manifest = prepare_data()
    print("Frozen cohort:", manifest["split_counts"])
    print("Split SHA-256:", manifest["split_sha256"])
