"""Feature engineering for stock forecasting."""

from __future__ import annotations

import pandas as pd


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend, spread, volatility, and lag features to price data."""
    data = df.copy()

    data["return_1d"] = data["close"].pct_change()
    data["ma_7"] = data["close"].rolling(window=7).mean()
    data["ma_20"] = data["close"].rolling(window=20).mean()
    data["ma_50"] = data["close"].rolling(window=50).mean()
    data["volatility_7"] = data["return_1d"].rolling(window=7).std()
    data["high_low_spread"] = data["high"] - data["low"]
    data["close_open_spread"] = data["close"] - data["open"]
    data["lag_1"] = data["close"].shift(1)
    data["lag_2"] = data["close"].shift(2)
    data["lag_3"] = data["close"].shift(3)

    return data


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build engineered features and drop rows made incomplete by rolling windows."""
    data = add_price_features(df)
    return data.dropna().reset_index(drop=True)
