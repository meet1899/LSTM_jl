"""Evaluation helpers for validation and test reporting."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def inverse_transform_targets(values: np.ndarray, scaler) -> np.ndarray:
    """Bring scaled targets or predictions back to the original price space."""
    values_2d = np.asarray(values).reshape(-1, 1)
    return scaler.inverse_transform(values_2d).ravel()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics for stock forecasting."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    non_zero = y_true != 0
    if np.any(non_zero):
        mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    else:
        mape = float("nan")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
    }


def evaluate_split(model, x_data: np.ndarray, y_data: np.ndarray, target_scaler) -> dict[str, float]:
    """Predict on a split and return metrics in the original scale."""
    preds_scaled = model.predict(x_data, verbose=0).ravel()
    actuals_scaled = np.asarray(y_data).ravel()

    preds = inverse_transform_targets(preds_scaled, target_scaler)
    actuals = inverse_transform_targets(actuals_scaled, target_scaler)

    return regression_metrics(actuals, preds)
