"""Helpers for saving training artifacts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_pickle(obj, path: str | Path) -> None:
    """Serialize a Python object with pickle."""
    with Path(path).open("wb") as handle:
        pickle.dump(obj, handle)


def save_json(data: dict, path: str | Path) -> None:
    """Write a dictionary to disk as pretty-printed JSON."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
