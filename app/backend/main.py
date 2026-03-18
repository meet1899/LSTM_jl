"""FastAPI app for the stock predictor backend."""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Query
from fastapi import Request

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
from src.logging_config import get_logger

app = FastAPI(title="Stock Predictor API", version="0.1.0")
logger = get_logger("app.backend.main")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request_complete method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    logger.info("health_check")
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    logger.info("model_info_requested")
    return ModelInfoResponse(**get_model_info())


@app.get("/history", response_model=HistoryResponse)
def history(ticker: str = Query(...), lookback_days: int = Query(60, ge=10, le=365)) -> HistoryResponse:
    try:
        points = get_historical_points(ticker=ticker, lookback_days=lookback_days)
    except ValueError as exc:
        logger.warning("history_request_failed ticker=%s lookback_days=%s detail=%s", ticker, lookback_days, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("history_request_success ticker=%s lookback_days=%s points=%s", ticker, lookback_days, len(points))
    return HistoryResponse(ticker=ticker.upper(), points=points)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        payload = predict_next_day(request.ticker, request.lookback_days)
    except ValueError as exc:
        logger.warning(
            "predict_request_failed ticker=%s lookback_days=%s detail=%s",
            request.ticker,
            request.lookback_days,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info(
        "predict_request_success ticker=%s lookback_days=%s source=%s",
        request.ticker,
        request.lookback_days,
        payload["prediction_source"],
    )
    return PredictionResponse(**payload)


@app.post("/forecast", response_model=ForecastResponse, responses={400: {"model": ErrorResponse}})
def forecast(request: ForecastRequest) -> ForecastResponse:
    try:
        payload = forecast_prices(request.ticker, request.lookback_days, request.horizon)
    except ValueError as exc:
        logger.warning(
            "forecast_request_failed ticker=%s lookback_days=%s horizon=%s detail=%s",
            request.ticker,
            request.lookback_days,
            request.horizon,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info(
        "forecast_request_success ticker=%s lookback_days=%s horizon=%s source=%s",
        request.ticker,
        request.lookback_days,
        request.horizon,
        payload["prediction_source"],
    )
    return ForecastResponse(**payload)


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    logger.info("metrics_requested")
    return MetricsResponse(**get_metrics_payload())


@app.get("/comparison", response_model=ComparisonResponse)
def comparison(split: str = Query("test", pattern="^(val|test)$")) -> ComparisonResponse:
    logger.info("comparison_requested split=%s", split)
    return ComparisonResponse(**get_comparison_payload(split=split))
