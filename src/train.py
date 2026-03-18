"""Training data preparation entry points."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd

from src.baselines import (
    linear_regression_baseline,
    moving_average_baseline,
    naive_last_value_baseline,
)
from src.config import TrainingConfig
from src.data_loader import load_local_dataset
from src.evaluate import SplitEvaluation, evaluate_predictions, evaluate_split, format_metrics
from src.features import finalize_features
from src.logging_config import get_logger
from src.preprocessing import chronological_split, scale_splits_train_only
from src.sequence import create_sequences, create_sequences_with_past_context
from src.utils import ensure_dir, save_json, save_pickle

logger = get_logger("src.train")


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
    target_col: str = "close",
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


def build_lstm_model(input_shape: tuple[int, int], config: TrainingConfig):
    """Create the baseline LSTM used for validation-aware training."""
    keras = _require_tensorflow()

    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(config.lstm_units_1, return_sequences=True),
            keras.layers.Dropout(config.dropout_rate),
            keras.layers.LSTM(config.lstm_units_2),
            keras.layers.Dropout(config.dropout_rate),
            keras.layers.Dense(config.dense_units, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=optimizer, loss="mse")
    return model


def train_model(df: pd.DataFrame, config: TrainingConfig) -> tuple[object, object, PreparedTrainingData]:
    """Train the LSTM on train data and monitor validation loss."""
    keras = _require_tensorflow()
    set_random_seed(config.random_seed)
    logger.info(
        "training_started rows=%s lookback=%s features=%s epochs=%s batch_size=%s",
        len(df),
        config.lookback,
        len(config.feature_cols),
        config.epochs,
        config.batch_size,
    )
    models_dir = ensure_dir(config.models_dir)
    checkpoint_path = models_dir / config.checkpoint_name

    prepared = prepare_training_data(
        df=df,
        target_col=config.target_col,
        feature_cols=list(config.feature_cols),
        lookback=config.lookback,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )
    logger.info(
        "training_data_prepared train=%s val=%s test=%s",
        prepared.x_train.shape[0],
        prepared.x_val.shape[0],
        prepared.x_test.shape[0],
    )

    model = build_lstm_model(
        input_shape=(prepared.x_train.shape[1], prepared.x_train.shape[2]),
        config=config,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
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
    logger.info("training_completed best_checkpoint=%s epochs_ran=%s", checkpoint_path, len(history.history.get("loss", [])))

    return model, history, prepared


def save_training_artifacts(results: TrainingArtifacts, config: TrainingConfig) -> dict[str, Path]:
    """Persist the trained model, scalers, config, metrics, and predictions."""
    models_dir = ensure_dir(config.models_dir)
    results_dir = ensure_dir(config.results_dir)

    model_path = models_dir / "lstm_model.keras"
    feature_scaler_path = models_dir / "feature_scaler.pkl"
    target_scaler_path = models_dir / "target_scaler.pkl"
    metadata_path = models_dir / "metadata.json"
    metrics_path = results_dir / "metrics.json"
    predictions_path = results_dir / "predictions.json"

    results.model.save(model_path)
    save_pickle(results.prepared.feature_scaler, feature_scaler_path)
    save_pickle(results.prepared.target_scaler, target_scaler_path)

    metadata = {
        "target_col": config.target_col,
        "feature_cols": list(config.feature_cols),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    save_json(metadata, metadata_path)

    metrics_payload = {
        "lstm": {
            "val": results.val_results.metrics,
            "test": results.test_results.metrics,
        },
        "baselines": {
            name: {
                "val": split_results["val"].metrics,
                "test": split_results["test"].metrics,
            }
            for name, split_results in results.baseline_results.items()
        },
        "history": results.history,
    }
    save_json(metrics_payload, metrics_path)

    predictions_payload = {
        "val_actuals": results.val_results.actuals.tolist(),
        "val_predictions": results.val_results.predictions.tolist(),
        "test_actuals": results.test_results.actuals.tolist(),
        "test_predictions": results.test_results.predictions.tolist(),
    }
    save_json(predictions_payload, predictions_path)
    logger.info(
        "training_artifacts_saved model=%s metrics=%s predictions=%s",
        model_path,
        metrics_path,
        predictions_path,
    )

    return {
        "model": model_path,
        "feature_scaler": feature_scaler_path,
        "target_scaler": target_scaler_path,
        "metadata": metadata_path,
        "metrics": metrics_path,
        "predictions": predictions_path,
        "checkpoint": Path(config.models_dir) / config.checkpoint_name,
    }


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
    if config.target_col in prepared.feature_cols:
        target_feature_index = prepared.feature_cols.index(config.target_col)
    else:
        target_feature_index = config.baseline_target_feature_index

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
        target_feature_index=target_feature_index,
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
    csv_path: str | None = None,
    config: TrainingConfig | None = None,
    save_artifacts: bool = True,
) -> TrainingArtifacts:
    """Convenience entrypoint for loading CSV data and training the model."""
    config = config or TrainingConfig()
    logger.info("training_pipeline_started csv_path=%s", csv_path or config.raw_data_path)
    df = load_local_dataset(path=csv_path, config=config)
    df = finalize_features(df)
    results = train_and_evaluate(df=df, config=config)

    print("LSTM Validation:", format_metrics(results.val_results.metrics))
    print("LSTM Test:", format_metrics(results.test_results.metrics))
    for baseline_name, split_results in results.baseline_results.items():
        print(f"{baseline_name} Validation:", format_metrics(split_results["val"].metrics))
        print(f"{baseline_name} Test:", format_metrics(split_results["test"].metrics))

    if save_artifacts:
        artifact_paths = save_training_artifacts(results, config)
        for artifact_name, artifact_path in artifact_paths.items():
            print(f"Saved {artifact_name}:", artifact_path)
    logger.info(
        "training_pipeline_completed val_metrics=%s test_metrics=%s",
        results.val_results.metrics,
        results.test_results.metrics,
    )

    return results
