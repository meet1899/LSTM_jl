"""Data loading helpers."""

from __future__ import annotations

import pandas as pd


def load_stock_csv(path: str) -> pd.DataFrame:
    """Load stock data from CSV and sort it in chronological order."""
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df
