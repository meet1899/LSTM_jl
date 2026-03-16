"""Project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Central training settings for the time-series pipeline."""

    target_col: str = "close"
    feature_cols: tuple[str, ...] = (
        "open",
        "high",
        "low",
        "close",
        "volume",
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
    )
    baseline_target_feature_index: int = 0
    lookback: int = 60
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    epochs: int = 50
    batch_size: int = 32
    random_seed: int = 42
    early_stopping_patience: int = 8
    learning_rate: float = 0.001
    lstm_units_1: int = 64
    lstm_units_2: int = 64
    dense_units: int = 25
    dropout_rate: float = 0.2
    models_dir: Path = Path("models")
    results_dir: Path = Path("results")
    checkpoint_name: str = "best_lstm.keras"
