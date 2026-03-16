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
