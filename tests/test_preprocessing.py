from __future__ import annotations

import numpy as np

from src.preprocessing import chronological_split, scale_splits_train_only


def test_chronological_split_preserves_order(sample_stock_df) -> None:
    split = chronological_split(sample_stock_df, train_ratio=0.70, val_ratio=0.15)
    assert len(split.train) == 56
    assert len(split.val) == 12
    assert len(split.test) == 12
    assert split.train["date"].max() < split.val["date"].min()
    assert split.val["date"].max() < split.test["date"].min()


def test_scale_splits_fit_on_train_only(sample_stock_df) -> None:
    split = chronological_split(sample_stock_df, train_ratio=0.70, val_ratio=0.15)
    scaled = scale_splits_train_only(split, feature_cols=["close"], target_col="close")
    expected_mean = np.mean(split.train[["close"]].to_numpy(), axis=0)
    assert np.allclose(scaled.feature_scaler.mean_, expected_mean)
    assert scaled.train_x.shape[0] == len(split.train)
    assert scaled.val_x.shape[0] == len(split.val)
