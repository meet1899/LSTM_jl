"""Project configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Central training settings for the time-series pipeline."""

    target_col: str = "close"
    feature_cols: tuple[str, ...] = ("close",)
    lookback: int = 60
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    epochs: int = 20
    batch_size: int = 32
    random_seed: int = 42
    early_stopping_patience: int = 5
