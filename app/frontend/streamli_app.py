"""Streamlit dashboard for Version 1 of the stock predictor app."""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"


def fetch_json(path: str, method: str = "GET", payload: dict | None = None) -> dict:
    url = f"{API_BASE_URL}{path}"
    if method == "POST":
        response = requests.post(url, json=payload, timeout=30)
    else:
        response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="Stock Predictor V1", layout="wide")
st.title("Stock Predictor V1")
st.caption("Interactive dashboard for local stock forecasting experiments.")

model_info = fetch_json("/model-info")
tickers = model_info.get("tickers", ["MSFT"])

with st.sidebar:
    st.header("Controls")
    ticker = st.selectbox("Ticker", options=tickers, index=0)
    lookback_days = st.slider("Lookback days", min_value=10, max_value=180, value=60, step=5)
    st.caption(model_info["disclaimer"])

history_payload = fetch_json(f"/history?ticker={ticker}&lookback_days={lookback_days}")
prediction_payload = fetch_json(
    "/predict",
    method="POST",
    payload={"ticker": ticker, "lookback_days": lookback_days},
)
metrics_payload = fetch_json("/metrics")
comparison_payload = fetch_json("/comparison?split=test")

history_df = pd.DataFrame(history_payload["points"])
comparison_df = pd.DataFrame(comparison_payload["points"])

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Historical Price Chart")
    st.line_chart(history_df.set_index("date")["close"])

    st.subheader("Actual vs Predicted")
    if comparison_df.empty:
        st.info("Prediction comparison data is not available yet.")
    else:
        st.line_chart(comparison_df.set_index("index")[["actual", "predicted"]])

with col2:
    st.subheader("Prediction")
    st.metric("Latest Close", f"{prediction_payload['latest_close']:.2f}")
    st.metric("Next-Day Prediction", f"{prediction_payload['next_day_prediction']:.2f}")
    st.write(f"Prediction source: `{prediction_payload['prediction_source']}`")

    st.subheader("Baseline Comparison")
    baseline_df = pd.DataFrame(
        [
            {"baseline": name, "prediction": value}
            for name, value in prediction_payload["baselines"].items()
        ]
    )
    st.dataframe(baseline_df, use_container_width=True, hide_index=True)

    st.subheader("Metrics")
    lstm_test = metrics_payload.get("lstm", {}).get("test", {})
    if lstm_test:
        st.json(lstm_test)
    else:
        st.info("No saved metrics found yet.")

st.subheader("Test Split Baselines")
baseline_metric_rows = []
for baseline_name, split_payload in metrics_payload.get("baselines", {}).items():
    baseline_metric_rows.append(
        {
            "baseline": baseline_name,
            **split_payload.get("test", {}),
        }
    )
if baseline_metric_rows:
    st.dataframe(pd.DataFrame(baseline_metric_rows), use_container_width=True, hide_index=True)
else:
    st.info("No baseline metric summary is available yet.")

st.warning(model_info["disclaimer"])
