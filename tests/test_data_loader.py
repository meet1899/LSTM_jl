from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import pytest

from src.data_loader import (
    build_refresh_metadata,
    get_available_tickers_from_data,
    save_refresh_metadata,
    validate_stock_dataframe,
)


def test_validate_stock_dataframe_accepts_valid_data(sample_stock_df) -> None:
    validate_stock_dataframe(sample_stock_df)


def test_validate_stock_dataframe_rejects_missing_column(sample_stock_df) -> None:
    broken = sample_stock_df.drop(columns=["close"])
    with pytest.raises(ValueError):
        validate_stock_dataframe(broken)


def test_get_available_tickers_from_data(sample_stock_df) -> None:
    assert get_available_tickers_from_data(sample_stock_df) == ["MSFT"]


def test_refresh_metadata_round_trip(sample_stock_df) -> None:
    metadata = build_refresh_metadata(sample_stock_df)
    temp_dir = Path("test_artifacts_temp") / str(uuid.uuid4())
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = temp_dir / "refresh.json"
        save_refresh_metadata(metadata, path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["row_count"] == len(sample_stock_df)
        assert loaded["tickers"] == ["MSFT"]
    finally:
        shutil.rmtree(temp_dir.parent, ignore_errors=True)
