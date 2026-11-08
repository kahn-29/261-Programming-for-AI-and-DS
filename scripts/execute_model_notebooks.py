"""Execute notebooks sequentially and save real outputs; fail on the first error."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.data import REPORTS, write_json


def execute(paths, working_directory):
    status_path = REPORTS / "execution_status.json"
    previous = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else []
    status = {item["notebook"]: item for item in previous}
    for path in paths:
        print(f"EXECUTING {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        start = time.perf_counter()
        try:
            NotebookClient(notebook, timeout=1800, kernel_name="python3",
                           resources={"metadata": {"path": str(working_directory)}},
                           record_timing=True).execute()
        finally:
            nbformat.write(notebook, path)
        status[path.name] = {"notebook": path.name, "seconds": time.perf_counter() - start,
                       "executed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
                       "status": "passed", "working_directory": working_directory.relative_to(ROOT).as_posix()}
        print(f"PASSED {path.name}: {status[path.name]['seconds']:.1f}s", flush=True)
        write_json(status_path, [status[key] for key in sorted(status)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*")
    parser.add_argument("--from-models", action="store_true")
    args = parser.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    paths = [ROOT / "models" / name for name in args.notebooks] if args.notebooks else sorted((ROOT / "models").glob("0*.ipynb"))
    execute(paths, ROOT / "models" if args.from_models else ROOT)
