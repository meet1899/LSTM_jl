"""Training data preparation entry points."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import pandas as pd

from src.baselines import (
    linear_regression_baseline,
    moving_average_baseline,
    naive_last_value_baseline,
)
from src.config import TrainingConfig
from src.evaluate import SplitEvaluation, evaluate_predictions, evaluate_split, format_metrics
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
    """Model, history, split evaluations, and processed arrays from a full training run."""

    model: object
    history: dict[str, list[float]]
    prepared: PreparedTrainingData
    val_results: SplitEvaluation
    test_results: SplitEvaluation
    baseline_results: dict[str, dict[str, SplitEvaluation]]


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


def run_baselines(
    prepared: PreparedTrainingData,
    target_feature_index: int = 0,
) -> dict[str, dict[str, SplitEvaluation]]:
    """Evaluate naive, moving-average, and linear-regression baselines."""
    baseline_results: dict[str, dict[str, SplitEvaluation]] = {}

    naive_val_pred = naive_last_value_baseline(prepared.x_val, target_feature_index)
    naive_test_pred = naive_last_value_baseline(prepared.x_test, target_feature_index)
    baseline_results["naive_last_value"] = {
        "val": evaluate_predictions(prepared.y_val, naive_val_pred, prepared.target_scaler),
        "test": evaluate_predictions(prepared.y_test, naive_test_pred, prepared.target_scaler),
    }

    moving_val_pred = moving_average_baseline(prepared.x_val, target_feature_index)
    moving_test_pred = moving_average_baseline(prepared.x_test, target_feature_index)
    baseline_results["moving_average"] = {
        "val": evaluate_predictions(prepared.y_val, moving_val_pred, prepared.target_scaler),
        "test": evaluate_predictions(prepared.y_test, moving_test_pred, prepared.target_scaler),
    }

    linear_val_pred = linear_regression_baseline(prepared.x_train, prepared.y_train, prepared.x_val)
    linear_test_pred = linear_regression_baseline(prepared.x_train, prepared.y_train, prepared.x_test)
    baseline_results["linear_regression"] = {
        "val": evaluate_predictions(prepared.y_val, linear_val_pred, prepared.target_scaler),
        "test": evaluate_predictions(prepared.y_test, linear_test_pred, prepared.target_scaler),
    }

    return baseline_results


def train_and_evaluate(df: pd.DataFrame, config: TrainingConfig | None = None) -> TrainingArtifacts:
    """Run the full training workflow: train, validate, then test once."""
    config = config or TrainingConfig()

    model, history, prepared = train_model(df, config)

    val_results = evaluate_split(
        model=model,
        x_data=prepared.x_val,
        y_data=prepared.y_val,
        target_scaler=prepared.target_scaler,
    )
    test_results = evaluate_split(
        model=model,
        x_data=prepared.x_test,
        y_data=prepared.y_test,
        target_scaler=prepared.target_scaler,
    )
    baseline_results = run_baselines(
        prepared=prepared,
        target_feature_index=config.baseline_target_feature_index,
    )

    return TrainingArtifacts(
        model=model,
        history=history.history,
        prepared=prepared,
        val_results=val_results,
        test_results=test_results,
        baseline_results=baseline_results,
    )


def run_training_pipeline(
    csv_path: str,
    config: TrainingConfig | None = None,
) -> TrainingArtifacts:
    """Convenience entrypoint for loading CSV data and training the model."""
    config = config or TrainingConfig()
    df = pd.read_csv(csv_path)
    results = train_and_evaluate(df=df, config=config)

    print("LSTM Validation:", format_metrics(results.val_results.metrics))
    print("LSTM Test:", format_metrics(results.test_results.metrics))
    for baseline_name, split_results in results.baseline_results.items():
        print(f"{baseline_name} Validation:", format_metrics(split_results["val"].metrics))
        print(f"{baseline_name} Test:", format_metrics(split_results["test"].metrics))

    return results
