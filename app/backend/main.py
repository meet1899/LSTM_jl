"""FastAPI app for the stock predictor backend."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from app.backend.schemas import (
    ComparisonResponse,
    ErrorResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    HistoryResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.backend.services import (
    forecast_prices,
    get_comparison_payload,
    get_historical_points,
    get_metrics_payload,
    get_model_info,
    predict_next_day,
)

app = FastAPI(title="Stock Predictor API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**get_model_info())


@app.get("/history", response_model=HistoryResponse)
def history(ticker: str = Query(...), lookback_days: int = Query(60, ge=10, le=365)) -> HistoryResponse:
    try:
        points = get_historical_points(ticker=ticker, lookback_days=lookback_days)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return HistoryResponse(ticker=ticker.upper(), points=points)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        payload = predict_next_day(request.ticker, request.lookback_days)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionResponse(**payload)


@app.post("/forecast", response_model=ForecastResponse, responses={400: {"model": ErrorResponse}})
def forecast(request: ForecastRequest) -> ForecastResponse:
    try:
        payload = forecast_prices(request.ticker, request.lookback_days, request.horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ForecastResponse(**payload)


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(**get_metrics_payload())


@app.get("/comparison", response_model=ComparisonResponse)
def comparison(split: str = Query("test", pattern="^(val|test)$")) -> ComparisonResponse:
    return ComparisonResponse(**get_comparison_payload(split=split))
