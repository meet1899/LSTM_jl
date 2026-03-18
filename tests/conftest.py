"""Shared pytest fixtures for the project test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def sample_stock_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    base = np.linspace(100.0, 140.0, len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "open": base,
            "high": base + 1.5,
            "low": base - 1.5,
            "close": base + 0.5,
            "volume": np.arange(len(dates)) + 1_000,
            "Name": ["MSFT"] * len(dates),
        }
    )


@pytest.fixture
def scaled_target_scaler() -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(np.array([[10.0], [12.0], [14.0], [16.0]]))
    return scaler
