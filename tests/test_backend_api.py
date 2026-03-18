"""Basic backend API tests for the FastAPI app."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info() -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert "tickers" in payload
    assert "artifacts_ready" in payload
    assert "prediction_mode" in payload
    assert "data_refresh" in payload


def test_history() -> None:
    response = client.get("/history", params={"ticker": "MSFT", "lookback_days": 30})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "MSFT"
    assert len(payload["points"]) == 30


def test_predict() -> None:
    response = client.post("/predict", json={"ticker": "MSFT", "lookback_days": 60})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "MSFT"
    assert "next_day_prediction" in payload
    assert "prediction_source" in payload


def test_forecast() -> None:
    response = client.post("/forecast", json={"ticker": "MSFT", "lookback_days": 60, "horizon": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "MSFT"
    assert payload["horizon"] == 3
    assert len(payload["points"]) == 3


def test_metrics_and_comparison() -> None:
    metrics_response = client.get("/metrics")
    comparison_response = client.get("/comparison", params={"split": "test"})
    assert metrics_response.status_code == 200
    assert comparison_response.status_code == 200


def test_invalid_ticker_returns_400() -> None:
    response = client.post("/predict", json={"ticker": "INVALID", "lookback_days": 60})
    assert response.status_code == 400
