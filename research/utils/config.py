import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(path) -> Path:
    """Config paths are relative to the repository root, regardless of cwd."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_data_root(cli_value: str | None) -> Path:
    value = cli_value or os.environ.get("HAM10000_ROOT")
    if not value:
        raise SystemExit(
            "HAM10000 dataset root not given. Pass --data-root or set HAM10000_ROOT to the folder "
            "containing HAM10000_metadata.csv and the HAM10000_images_part_* folders."
        )
    root = Path(value)
    if not root.is_dir():
        raise SystemExit(f"HAM10000 dataset root does not exist: {root}")
    return root
