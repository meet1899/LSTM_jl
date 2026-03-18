from __future__ import annotations

import numpy as np

from src.evaluate import direction_accuracy, evaluate_predictions, regression_metrics


def test_regression_metrics_contains_expected_keys() -> None:
    metrics = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 2.5]))
    assert {"mae", "rmse", "mape", "direction_accuracy"} == set(metrics.keys())


def test_direction_accuracy_matches_trend_changes() -> None:
    score = direction_accuracy(
        np.array([10.0, 11.0, 10.0, 12.0]),
        np.array([9.0, 10.0, 9.5, 11.0]),
    )
    assert score == 1.0


def test_evaluate_predictions_returns_split_evaluation(scaled_target_scaler) -> None:
    result = evaluate_predictions(
        y_true_scaled=np.array([[-1.0], [0.0], [1.0]]),
        y_pred_scaled=np.array([[-0.8], [0.2], [0.9]]),
        target_scaler=scaled_target_scaler,
    )
    assert result.actuals.shape == (3,)
    assert result.predictions.shape == (3,)
    assert "mae" in result.metrics
