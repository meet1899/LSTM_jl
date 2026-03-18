from __future__ import annotations

import numpy as np

from src.baselines import (
    linear_regression_baseline,
    moving_average_baseline,
    naive_last_value_baseline,
)


def test_naive_last_value_baseline_returns_last_value() -> None:
    x_data = np.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    preds = naive_last_value_baseline(x_data)
    assert np.allclose(preds, np.array([3.0, 6.0]))


def test_moving_average_baseline_returns_window_mean() -> None:
    x_data = np.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    preds = moving_average_baseline(x_data)
    assert np.allclose(preds, np.array([2.0, 5.0]))


def test_linear_regression_baseline_shape() -> None:
    x_train = np.arange(24, dtype=float).reshape(4, 3, 2)
    y_train = np.array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
    x_eval = np.arange(12, dtype=float).reshape(2, 3, 2)
    preds = linear_regression_baseline(x_train, y_train, x_eval)
    assert preds.shape == (2,)
