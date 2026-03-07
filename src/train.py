"""Training data preparation entry points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.preprocessing import chronological_split, scale_splits_train_only
from src.sequence import create_sequences, create_sequences_with_past_context


@dataclass(frozen=True)
class PreparedTrainingData:
    """Leakage-safe arrays and artifacts needed for model training/evaluation."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    y_test_raw: np.ndarray
    feature_cols: list[str]
    target_col: str
    feature_scaler: object
    target_scaler: object


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = "Close",
    feature_cols: list[str] | None = None,
    lookback: int = 60,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> PreparedTrainingData:
    """
    Build train/val/test datasets without data leakage.

    Steps:
    1) Chronological split
    2) Fit scalers on train only
    3) Transform val/test using train-fitted scalers
    4) Build sequences with past-only context
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe")

    if feature_cols is None:
        feature_cols = [target_col]

    split = chronological_split(df, train_ratio=train_ratio, val_ratio=val_ratio)
    scaled = scale_splits_train_only(split, feature_cols=feature_cols, target_col=target_col)

    x_train, y_train = create_sequences(scaled.train_x, scaled.train_y, lookback=lookback)

    x_val, y_val = create_sequences_with_past_context(
        features=scaled.val_x,
        targets=scaled.val_y,
        lookback=lookback,
        past_features=scaled.train_x,
        past_targets=scaled.train_y,
    )

    train_val_x = np.vstack([scaled.train_x, scaled.val_x])
    train_val_y = np.vstack([scaled.train_y, scaled.val_y])

    x_test, y_test = create_sequences_with_past_context(
        features=scaled.test_x,
        targets=scaled.test_y,
        lookback=lookback,
        past_features=train_val_x,
        past_targets=train_val_y,
    )

    y_test_raw = split.test[target_col].to_numpy()

    return PreparedTrainingData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        y_test_raw=y_test_raw,
        feature_cols=feature_cols,
        target_col=target_col,
        feature_scaler=scaled.feature_scaler,
        target_scaler=scaled.target_scaler,
    )
