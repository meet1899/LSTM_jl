from __future__ import annotations

import numpy as np

from src.train import prepare_training_data, run_baselines


def test_prepare_training_data_returns_expected_shapes(sample_stock_df) -> None:
    prepared = prepare_training_data(
        sample_stock_df,
        target_col="close",
        feature_cols=["close"],
        lookback=5,
        train_ratio=0.70,
        val_ratio=0.15,
    )
    assert prepared.x_train.ndim == 3
    assert prepared.x_val.ndim == 3
    assert prepared.x_test.ndim == 3
    assert prepared.x_train.shape[1] == 5


def test_run_baselines_returns_all_expected_models(sample_stock_df) -> None:
    prepared = prepare_training_data(
        sample_stock_df,
        target_col="close",
        feature_cols=["close"],
        lookback=5,
        train_ratio=0.70,
        val_ratio=0.15,
    )
    baseline_results = run_baselines(prepared, target_feature_index=0)
    assert {"naive_last_value", "moving_average", "linear_regression"} == set(baseline_results.keys())
    assert "test" in baseline_results["naive_last_value"]
