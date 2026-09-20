from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("EM_DATA_DIR", Path.home() / ".cache" / "em-distributional-transition")).expanduser()
OUTPUT_DIR = Path(os.getenv("EM_OUTPUT_DIR", "outputs")).expanduser()

DATASETS_DIR = DATA_DIR / "datasets"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
FIGURES_DIR = OUTPUT_DIR / "figures"


def run_artifacts(run_id: str) -> Path:
    """Return the artifact directory for a given run ID."""
    return ARTIFACTS_DIR / run_id


def ensure_dirs() -> None:
    """Create all data and output directories."""
    for d in (DATASETS_DIR, ARTIFACTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    dirs = {
        "DATA_DIR": DATA_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "DATASETS_DIR": DATASETS_DIR,
        "ARTIFACTS_DIR": ARTIFACTS_DIR,
        "FIGURES_DIR": FIGURES_DIR,
    }
    for name, path in dirs.items():
        n = len(list(path.glob("*"))) if path.exists() else 0
        print(f"{name:<15} {path}  ({n} items)")
