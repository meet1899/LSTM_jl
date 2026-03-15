"""Baseline forecasting models for comparison against the LSTM."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def naive_last_value_baseline(x_data: np.ndarray, target_feature_index: int = 0) -> np.ndarray:
    """Predict the next value as the last observed value in the input window."""
    return x_data[:, -1, target_feature_index]


def moving_average_baseline(x_data: np.ndarray, target_feature_index: int = 0) -> np.ndarray:
    """Predict the next value as the average of the input window."""
    return np.mean(x_data[:, :, target_feature_index], axis=1)


def linear_regression_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """Train a simple linear regression model on flattened windows."""
    model = LinearRegression()
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_eval_flat = x_eval.reshape(x_eval.shape[0], -1)
    model.fit(x_train_flat, y_train.ravel())
    return model.predict(x_eval_flat)
