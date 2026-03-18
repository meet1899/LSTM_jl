"""Pydantic schemas for the stock predictor API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str


class ModelInfoResponse(BaseModel):
    model_name: str
    target_col: str
    feature_cols: list[str]
    tickers: list[str]
    available_baselines: list[str]
    artifacts_ready: bool
    prediction_mode: str
    data_refresh: dict[str, object]
    disclaimer: str


class PredictionRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    lookback_days: int = Field(default=60, ge=10, le=200)


class PredictionResponse(BaseModel):
    ticker: str
    lookback_days: int
    latest_close: float
    next_day_prediction: float
    baselines: dict[str, float]
    prediction_source: str
    disclaimer: str


class ForecastRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    lookback_days: int = Field(default=60, ge=10, le=200)
    horizon: int = Field(default=5, ge=1, le=30)


class ForecastPoint(BaseModel):
    step: int
    predicted_close: float


class ForecastResponse(BaseModel):
    ticker: str
    lookback_days: int
    horizon: int
    points: list[ForecastPoint]
    baselines: dict[str, list[float]]
    prediction_source: str
    disclaimer: str


class HistoricalPoint(BaseModel):
    date: str
    close: float
    volume: float


class HistoryResponse(BaseModel):
    ticker: str
    points: list[HistoricalPoint]


class MetricsResponse(BaseModel):
    lstm: dict[str, dict[str, float]]
    baselines: dict[str, dict[str, dict[str, float]]]


class ComparisonPoint(BaseModel):
    index: int
    actual: float
    predicted: float


class ComparisonResponse(BaseModel):
    split: str
    points: list[ComparisonPoint]
