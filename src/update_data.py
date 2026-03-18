"""Utilities for refreshing local dataset metadata."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import TrainingConfig
from src.data_loader import build_refresh_metadata, load_local_dataset, save_refresh_metadata


def refresh_local_data_metadata(config: TrainingConfig | None = None) -> dict[str, object]:
    """Validate the local dataset and refresh the metadata snapshot."""
    config = config or TrainingConfig()
    df = load_local_dataset(config=config)
    metadata = build_refresh_metadata(df, source_type="csv")
    save_refresh_metadata(metadata, config.refresh_metadata_path)
    return metadata


if __name__ == "__main__":
    refreshed = refresh_local_data_metadata()
    print(refreshed)
