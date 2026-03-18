from __future__ import annotations

from src.features import add_price_features, finalize_features


def test_add_price_features_creates_expected_columns(sample_stock_df) -> None:
    featured = add_price_features(sample_stock_df)
    expected = {
        "return_1d",
        "ma_7",
        "ma_20",
        "ma_50",
        "volatility_7",
        "high_low_spread",
        "close_open_spread",
        "lag_1",
        "lag_2",
        "lag_3",
    }
    assert expected.issubset(set(featured.columns))


def test_finalize_features_drops_nan_rows(sample_stock_df) -> None:
    featured = finalize_features(sample_stock_df)
    assert not featured.isna().any().any()
    assert len(featured) < len(sample_stock_df)
