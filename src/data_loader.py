"""Data loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import TrainingConfig
from src.logging_config import get_logger

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}
logger = get_logger("src.data_loader")


def validate_stock_dataframe(df: pd.DataFrame) -> None:
    """Validate the minimum schema expected by the forecasting pipeline."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Loaded stock dataframe is empty.")

    if df["date"].isna().any():
        raise ValueError("The dataset contains missing dates.")

    duplicate_cols = ["date"]
    if "Name" in df.columns:
        duplicate_cols.append("Name")
    if df.duplicated(subset=duplicate_cols).any():
        raise ValueError(f"Duplicate rows found for key columns: {duplicate_cols}")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    if df[numeric_cols].isna().any().any():
        raise ValueError("The dataset contains missing numeric price/volume values.")
    logger.info("stock_dataframe_validated rows=%s columns=%s", len(df), sorted(df.columns.tolist()))


def load_stock_csv(path: str) -> pd.DataFrame:
    """Load stock data from CSV and sort it in chronological order."""
    logger.info("loading_stock_csv path=%s", path)
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    validate_stock_dataframe(df)
    return df


def load_local_dataset(path: str | Path | None = None, config: TrainingConfig | None = None) -> pd.DataFrame:
    """Load the canonical local dataset using config defaults when path is omitted."""
    config = config or TrainingConfig()
    resolved_path = Path(path) if path is not None else config.raw_data_path
    return load_stock_csv(str(resolved_path))


def get_available_tickers_from_data(df: pd.DataFrame, default_ticker: str = "MSFT") -> list[str]:
    """Return sorted ticker symbols from the local dataset."""
    if "Name" in df.columns:
        return sorted(df["Name"].dropna().astype(str).str.upper().unique().tolist())
    return [default_ticker]


def save_stock_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a validated dataset to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("stock_csv_saved path=%s rows=%s", path, len(df))
    return path


def build_refresh_metadata(df: pd.DataFrame, source_type: str = "csv") -> dict[str, object]:
    """Create metadata describing the current local dataset snapshot."""
    tickers = get_available_tickers_from_data(df)
    metadata = {
        "source_type": source_type,
        "row_count": int(len(df)),
        "tickers": tickers,
        "min_date": df["date"].min().strftime("%Y-%m-%d"),
        "max_date": df["date"].max().strftime("%Y-%m-%d"),
        "last_refresh_utc": pd.Timestamp.utcnow().isoformat(),
    }
    return metadata


def save_refresh_metadata(metadata: dict[str, object], path: str | Path) -> Path:
    """Persist refresh metadata to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("refresh_metadata_saved path=%s", path)
    return path


def load_refresh_metadata(path: str | Path, default: dict[str, object] | None = None) -> dict[str, object]:
    """Load refresh metadata if present, otherwise return a default object."""
    path = Path(path)
    if not path.exists():
        return default or {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
