"""Evaluation helpers for validation and test reporting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class SplitEvaluation:
    """Evaluation payload for a single split."""

    metrics: dict[str, float]
    actuals: np.ndarray
    predictions: np.ndarray


def inverse_transform_targets(values: np.ndarray, scaler) -> np.ndarray:
    """Bring scaled targets or predictions back to the original price space."""
    values_2d = np.asarray(values).reshape(-1, 1)
    return scaler.inverse_transform(values_2d).ravel()


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Measure how often the forecast gets the movement direction right."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) < 2 or len(y_pred) < 2:
        return float("nan")

    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    return float(np.mean(true_direction == pred_direction))


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
    dir_acc = direction_accuracy(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "direction_accuracy": float(dir_acc),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    """Render metrics in a compact, readable form."""
    return (
        f"MAE: {metrics['mae']:.4f}, "
        f"RMSE: {metrics['rmse']:.4f}, "
        f"MAPE: {metrics['mape']:.2f}%, "
        f"Direction Accuracy: {metrics['direction_accuracy']:.2%}"
    )


def evaluate_split(model, x_data: np.ndarray, y_data: np.ndarray, target_scaler) -> SplitEvaluation:
    """Predict on a split and return both metrics and original-scale outputs."""
    preds_scaled = model.predict(x_data, verbose=0).ravel()
    actuals_scaled = np.asarray(y_data).ravel()

    return evaluate_predictions(actuals_scaled, preds_scaled, target_scaler)


def evaluate_predictions(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    target_scaler,
) -> SplitEvaluation:
    """Evaluate precomputed scaled predictions using the shared metric pipeline."""
    preds = inverse_transform_targets(y_pred_scaled, target_scaler)
    actuals = inverse_transform_targets(y_true_scaled, target_scaler)

    return SplitEvaluation(
        metrics=regression_metrics(actuals, preds),
        actuals=actuals,
        predictions=preds,
    )
