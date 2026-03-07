"""Scaling and chronological split utilities for leakage-safe time series training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ChronologicalSplit:
    """Container for chronological train/val/test splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class ScaledSplit:
    """Container for scaled split arrays and fitted scalers."""

    train_x: np.ndarray
    val_x: np.ndarray
    test_x: np.ndarray
    train_y: np.ndarray
    val_y: np.ndarray
    test_y: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler


def chronological_split(
    df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> ChronologicalSplit:
    """Split a time-ordered dataframe into train/val/test without shuffling."""
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1.")
    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if train_end < 2 or val_end <= train_end or val_end >= n:
        raise ValueError(
            "Not enough rows for train/val/test split. Add more data or adjust split ratios."
        )

    return ChronologicalSplit(
        train=df.iloc[:train_end].copy(),
        val=df.iloc[train_end:val_end].copy(),
        test=df.iloc[val_end:].copy(),
    )


def scale_splits_train_only(
    split: ChronologicalSplit, feature_cols: list[str], target_col: str
) -> ScaledSplit:
    """Fit scalers on train split only, then transform val/test."""
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one column.")
    missing = [c for c in feature_cols + [target_col] if c not in split.train.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    train_x = x_scaler.fit_transform(split.train[feature_cols].to_numpy())
    val_x = x_scaler.transform(split.val[feature_cols].to_numpy())
    test_x = x_scaler.transform(split.test[feature_cols].to_numpy())

    train_y = y_scaler.fit_transform(split.train[[target_col]].to_numpy())
    val_y = y_scaler.transform(split.val[[target_col]].to_numpy())
    test_y = y_scaler.transform(split.test[[target_col]].to_numpy())

    assert_no_scaler_leakage(x_scaler, split.train[feature_cols].to_numpy())
    assert_no_scaler_leakage(y_scaler, split.train[[target_col]].to_numpy())

    return ScaledSplit(
        train_x=train_x,
        val_x=val_x,
        test_x=test_x,
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        feature_scaler=x_scaler,
        target_scaler=y_scaler,
    )


def assert_no_scaler_leakage(
    scaler: StandardScaler, train_values: np.ndarray, atol: float = 1e-12
) -> None:
    """Sanity check that fitted scaler statistics match train-only statistics."""
    if hasattr(scaler, "mean_"):
        expected = np.mean(train_values, axis=0)
        if not np.allclose(scaler.mean_, expected, atol=atol):
            raise AssertionError("Scaler mean does not match train-only statistics.")
