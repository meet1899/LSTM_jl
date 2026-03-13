"""Training data preparation entry points."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import pandas as pd

from src.config import TrainingConfig
from src.evaluate import evaluate_split
from src.preprocessing import chronological_split, scale_splits_train_only
from src.sequence import create_sequences, create_sequences_with_past_context


@dataclass(frozen=True)
class PreparedTrainingData:
    """Leakage-safe arrays and artifacts needed for model training/evaluation."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    y_test_raw: np.ndarray
    feature_cols: list[str]
    target_col: str
    feature_scaler: object
    target_scaler: object


@dataclass(frozen=True)
class TrainingArtifacts:
    """Model, history, metrics, and processed arrays from a full training run."""

    model: object
    history: dict[str, list[float]]
    prepared: PreparedTrainingData
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = "Close",
    feature_cols: list[str] | None = None,
    lookback: int = 60,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> PreparedTrainingData:
    """
    Build train/val/test datasets without data leakage.

    Steps:
    1) Chronological split
    2) Fit scalers on train only
    3) Transform val/test using train-fitted scalers
    4) Build sequences with past-only context
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe")

    if feature_cols is None:
        feature_cols = [target_col]

    split = chronological_split(df, train_ratio=train_ratio, val_ratio=val_ratio)
    scaled = scale_splits_train_only(split, feature_cols=feature_cols, target_col=target_col)

    x_train, y_train = create_sequences(scaled.train_x, scaled.train_y, lookback=lookback)

    x_val, y_val = create_sequences_with_past_context(
        features=scaled.val_x,
        targets=scaled.val_y,
        lookback=lookback,
        past_features=scaled.train_x,
        past_targets=scaled.train_y,
    )

    train_val_x = np.vstack([scaled.train_x, scaled.val_x])
    train_val_y = np.vstack([scaled.train_y, scaled.val_y])

    x_test, y_test = create_sequences_with_past_context(
        features=scaled.test_x,
        targets=scaled.test_y,
        lookback=lookback,
        past_features=train_val_x,
        past_targets=train_val_y,
    )

    y_test_raw = split.test[target_col].to_numpy()

    return PreparedTrainingData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        y_test_raw=y_test_raw,
        feature_cols=feature_cols,
        target_col=target_col,
        feature_scaler=scaled.feature_scaler,
        target_scaler=scaled.target_scaler,
    )


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow when available."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        return

    tf.random.set_seed(seed)


def _require_tensorflow():
    try:
        from tensorflow import keras
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required for model training. Install it before running Step 3B training."
        ) from exc
    return keras


def build_lstm_model(input_shape: tuple[int, int]):
    """Create the baseline LSTM used for validation-aware training."""
    keras = _require_tensorflow()

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.LSTM(64),
            keras.layers.Dense(25, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_model(df: pd.DataFrame, config: TrainingConfig) -> tuple[object, object, PreparedTrainingData]:
    """Train the LSTM on train data and monitor validation loss."""
    keras = _require_tensorflow()
    set_random_seed(config.random_seed)

    prepared = prepare_training_data(
        df=df,
        target_col=config.target_col,
        feature_cols=list(config.feature_cols),
        lookback=config.lookback,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    model = build_lstm_model(
        input_shape=(prepared.x_train.shape[1], prepared.x_train.shape[2])
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        prepared.x_train,
        prepared.y_train,
        validation_data=(prepared.x_val, prepared.y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    return model, history, prepared


def train_and_evaluate(df: pd.DataFrame, config: TrainingConfig | None = None) -> TrainingArtifacts:
    """Run the full Step 3B workflow: train, validate, then test once."""
    config = config or TrainingConfig()

    model, history, prepared = train_model(df, config)

    val_metrics = evaluate_split(
        model=model,
        x_data=prepared.x_val,
        y_data=prepared.y_val,
        target_scaler=prepared.target_scaler,
    )
    test_metrics = evaluate_split(
        model=model,
        x_data=prepared.x_test,
        y_data=prepared.y_test,
        target_scaler=prepared.target_scaler,
    )

    return TrainingArtifacts(
        model=model,
        history=history.history,
        prepared=prepared,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )


def run_training_pipeline(
    csv_path: str,
    config: TrainingConfig | None = None,
) -> TrainingArtifacts:
    """Convenience entrypoint for loading CSV data and training the model."""
    config = config or TrainingConfig()
    df = pd.read_csv(csv_path)
    return train_and_evaluate(df=df, config=config)
