"""Backend services for the stock predictor app."""

from __future__ import annotations

from functools import lru_cache
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TrainingConfig
from src.data_loader import load_stock_csv
from src.features import finalize_features

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "MicrosoftStock.csv"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
DISCLAIMER = "For educational use only. This app does not provide financial advice."
AVAILABLE_BASELINES = ["naive_last_value", "moving_average", "linear_regression"]


def _load_base_dataframe() -> pd.DataFrame:
    return load_stock_csv(str(DATA_PATH))


def get_available_tickers() -> list[str]:
    df = _load_base_dataframe()
    if "Name" in df.columns:
        return sorted(df["Name"].dropna().unique().tolist())
    return ["MSFT"]


def _get_ticker_frame(ticker: str) -> pd.DataFrame:
    df = _load_base_dataframe()
    if "Name" in df.columns:
        filtered = df[df["Name"].str.upper() == ticker.upper()].copy()
    else:
        filtered = df.copy()
    if filtered.empty:
        raise ValueError(f"Ticker '{ticker}' is not available in the local dataset.")
    return filtered.reset_index(drop=True)


def get_historical_points(ticker: str, lookback_days: int) -> list[dict[str, float | str]]:
    df = _get_ticker_frame(ticker).tail(lookback_days)
    return [
        {
            "date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        for _, row in df.iterrows()
    ]


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_keras_model(path: Path):
    from tensorflow import keras

    return keras.models.load_model(path)


def _resolve_model_path() -> Path | None:
    best = MODELS_DIR / "best_lstm.keras"
    final = MODELS_DIR / "lstm_model.keras"
    if best.exists():
        return best
    if final.exists():
        return final
    return None


def _latest_feature_row(ticker: str, config: TrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = _get_ticker_frame(ticker)
    feature_df = finalize_features(raw_df)
    return raw_df, feature_df


@lru_cache(maxsize=1)
def load_prediction_artifacts() -> dict[str, object] | None:
    model_path = _resolve_model_path()
    feature_scaler_path = MODELS_DIR / "feature_scaler.pkl"
    target_scaler_path = MODELS_DIR / "target_scaler.pkl"

    if not model_path or not feature_scaler_path.exists() or not target_scaler_path.exists():
        return None

    try:
        return {
            "model": _load_keras_model(model_path),
            "feature_scaler": _load_pickle(feature_scaler_path),
            "target_scaler": _load_pickle(target_scaler_path),
            "model_path": model_path,
        }
    except Exception:
        return None


def artifacts_ready() -> bool:
    return load_prediction_artifacts() is not None


def _prepare_recent_window(ticker: str, lookback_days: int, config: TrainingConfig) -> tuple[pd.DataFrame, np.ndarray]:
    raw_df, feature_df = _latest_feature_row(ticker, config)
    if len(feature_df) < lookback_days:
        raise ValueError(
            f"Not enough feature rows for lookback_days={lookback_days}. Available rows: {len(feature_df)}."
        )
    recent_window = feature_df[list(config.feature_cols)].tail(lookback_days).to_numpy()
    return raw_df, recent_window


def _compute_baseline_predictions(recent_window: np.ndarray, target_index: int, horizon: int) -> dict[str, list[float]]:
    naive_value = float(recent_window[-1, target_index])
    moving_average_value = float(np.mean(recent_window[:, target_index]))
    return {
        "naive_last_value": [naive_value for _ in range(horizon)],
        "moving_average": [moving_average_value for _ in range(horizon)],
    }


def _predict_with_model(recent_window: np.ndarray) -> float:
    artifacts = load_prediction_artifacts()
    if artifacts is None:
        raise RuntimeError("Prediction artifacts are not ready.")

    scaled_window = artifacts["feature_scaler"].transform(recent_window)
    x_input = np.expand_dims(scaled_window, axis=0)
    pred_scaled = artifacts["model"].predict(x_input, verbose=0).ravel()[0]
    return float(artifacts["target_scaler"].inverse_transform([[pred_scaled]])[0, 0])


def predict_next_day(ticker: str, lookback_days: int) -> dict[str, object]:
    config = TrainingConfig()
    raw_df, recent_window = _prepare_recent_window(ticker, lookback_days, config)
    latest_close = float(raw_df.iloc[-1]["close"])
    target_index = list(config.feature_cols).index(config.target_col)
    baseline_predictions = _compute_baseline_predictions(recent_window, target_index, horizon=1)
    baselines = {name: values[0] for name, values in baseline_predictions.items()}

    if not artifacts_ready():
        return {
            "ticker": ticker.upper(),
            "lookback_days": lookback_days,
            "latest_close": latest_close,
            "next_day_prediction": baselines["naive_last_value"],
            "baselines": baselines,
            "prediction_source": "naive_last_value_fallback",
            "disclaimer": DISCLAIMER,
        }

    try:
        next_day_prediction = _predict_with_model(recent_window)
    except Exception:
        return {
            "ticker": ticker.upper(),
            "lookback_days": lookback_days,
            "latest_close": latest_close,
            "next_day_prediction": baselines["naive_last_value"],
            "baselines": baselines,
            "prediction_source": "naive_last_value_fallback",
            "disclaimer": DISCLAIMER,
        }

    return {
        "ticker": ticker.upper(),
        "lookback_days": lookback_days,
        "latest_close": latest_close,
        "next_day_prediction": next_day_prediction,
        "baselines": baselines,
        "prediction_source": "trained_lstm",
        "disclaimer": DISCLAIMER,
    }


def forecast_prices(ticker: str, lookback_days: int, horizon: int) -> dict[str, object]:
    config = TrainingConfig()
    _, recent_window = _prepare_recent_window(ticker, lookback_days, config)
    target_index = list(config.feature_cols).index(config.target_col)
    baseline_predictions = _compute_baseline_predictions(recent_window, target_index, horizon=horizon)

    if not artifacts_ready():
        return {
            "ticker": ticker.upper(),
            "lookback_days": lookback_days,
            "horizon": horizon,
            "points": [
                {"step": step + 1, "predicted_close": baseline_predictions["naive_last_value"][step]}
                for step in range(horizon)
            ],
            "baselines": baseline_predictions,
            "prediction_source": "naive_last_value_fallback",
            "disclaimer": DISCLAIMER,
        }

    rolling_window = recent_window.copy()
    points = []
    try:
        for step in range(horizon):
            predicted_close = _predict_with_model(rolling_window)
            points.append({"step": step + 1, "predicted_close": predicted_close})
            next_row = rolling_window[-1].copy()
            next_row[target_index] = predicted_close
            rolling_window = np.vstack([rolling_window[1:], next_row])
    except Exception:
        return {
            "ticker": ticker.upper(),
            "lookback_days": lookback_days,
            "horizon": horizon,
            "points": [
                {"step": step + 1, "predicted_close": baseline_predictions["naive_last_value"][step]}
                for step in range(horizon)
            ],
            "baselines": baseline_predictions,
            "prediction_source": "naive_last_value_fallback",
            "disclaimer": DISCLAIMER,
        }

    return {
        "ticker": ticker.upper(),
        "lookback_days": lookback_days,
        "horizon": horizon,
        "points": points,
        "baselines": baseline_predictions,
        "prediction_source": "trained_lstm",
        "disclaimer": DISCLAIMER,
    }


def get_metrics_payload() -> dict[str, object]:
    metrics_path = RESULTS_DIR / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {
            "lstm": payload.get("lstm", {}),
            "baselines": payload.get("baselines", {}),
        }

    return {
        "lstm": {"val": {}, "test": {}},
        "baselines": {},
    }


def get_comparison_payload(split: str = "test") -> dict[str, object]:
    predictions_path = RESULTS_DIR / "predictions.json"
    if not predictions_path.exists():
        return {"split": split, "points": []}

    with predictions_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    actuals = payload.get(f"{split}_actuals", [])
    predictions = payload.get(f"{split}_predictions", [])
    points = [
        {
            "index": idx,
            "actual": float(actual),
            "predicted": float(pred),
        }
        for idx, (actual, pred) in enumerate(zip(actuals, predictions))
    ]
    return {"split": split, "points": points}


def get_model_info() -> dict[str, object]:
    config = TrainingConfig()
    return {
        "model_name": "LSTM Stock Predictor",
        "target_col": config.target_col,
        "feature_cols": list(config.feature_cols),
        "tickers": get_available_tickers(),
        "available_baselines": AVAILABLE_BASELINES,
        "artifacts_ready": artifacts_ready(),
        "prediction_mode": "trained_lstm" if artifacts_ready() else "naive_last_value_fallback",
        "disclaimer": DISCLAIMER,
    }
