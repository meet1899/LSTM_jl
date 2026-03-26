"""Streamlit dashboard for the stock predictor app."""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def fetch_json(path: str, method: str = "GET", payload: dict | None = None) -> dict:
    """Fetch JSON from the backend API and surface readable UI errors."""
    url = f"{API_BASE_URL}{path}"
    try:
        if method == "POST":
            response = requests.post(url, json=payload, timeout=30)
        else:
            response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Backend request failed for `{path}`: {exc}")
        st.stop()


def metric_card_columns(metrics: dict[str, float]) -> None:
    """Render the core evaluation metrics as Streamlit metric cards."""
    if not metrics:
        st.info("No saved metrics are available yet.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{metrics.get('mae', 0.0):.4f}")
    col2.metric("RMSE", f"{metrics.get('rmse', 0.0):.4f}")
    col3.metric("MAPE", f"{metrics.get('mape', 0.0):.2f}%")
    col4.metric("Direction Accuracy", f"{metrics.get('direction_accuracy', 0.0):.2%}")


def build_baseline_prediction_table(baselines: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"baseline": name, "prediction": value} for name, value in baselines.items()]
    )


def build_baseline_metric_table(metrics_payload: dict[str, object], split: str) -> pd.DataFrame:
    rows = []
    for baseline_name, split_payload in metrics_payload.get("baselines", {}).items():
        rows.append(
            {
                "baseline": baseline_name,
                **split_payload.get(split, {}),
            }
        )
    return pd.DataFrame(rows)


def build_forecast_frame(forecast_payload: dict[str, object]) -> pd.DataFrame:
    points = forecast_payload.get("points", [])
    return pd.DataFrame(points)


def main() -> None:
    st.set_page_config(page_title="Stock Predictor Dashboard", layout="wide")
    st.title("Stock Predictor Dashboard")
    st.caption("Interactive forecasting dashboard powered by the local FastAPI backend.")

    model_info = fetch_json("/model-info")
    tickers = model_info.get("tickers", ["MSFT"])

    with st.sidebar:
        st.header("Controls")
        ticker = st.selectbox("Ticker", options=tickers, index=0)
        lookback_days = st.slider("Lookback days", min_value=10, max_value=180, value=60, step=5)
        forecast_horizon = st.slider("Forecast horizon", min_value=1, max_value=10, value=3, step=1)
        comparison_split = st.radio("Comparison split", options=["test", "val"], horizontal=True)

        st.header("Model Status")
        st.write(f"Prediction mode: `{model_info['prediction_mode']}`")
        st.write(f"Artifacts ready: `{model_info['artifacts_ready']}`")
        st.write("Available baselines:")
        st.write(", ".join(model_info.get("available_baselines", [])) or "None")
        st.caption(model_info["disclaimer"])

    history_payload = fetch_json(f"/history?ticker={ticker}&lookback_days={lookback_days}")
    prediction_payload = fetch_json(
        "/predict",
        method="POST",
        payload={"ticker": ticker, "lookback_days": lookback_days},
    )
    forecast_payload = fetch_json(
        "/forecast",
        method="POST",
        payload={
            "ticker": ticker,
            "lookback_days": lookback_days,
            "horizon": forecast_horizon,
        },
    )
    metrics_payload = fetch_json("/metrics")
    comparison_payload = fetch_json(f"/comparison?split={comparison_split}")

    history_df = pd.DataFrame(history_payload["points"])
    comparison_df = pd.DataFrame(comparison_payload["points"])
    forecast_df = build_forecast_frame(forecast_payload)

    hero_left, hero_right = st.columns([2, 1])
    with hero_left:
        st.subheader("Historical Price")
        history_chart_df = history_df.copy()
        if not history_chart_df.empty:
            history_chart_df["date"] = pd.to_datetime(history_chart_df["date"])
            st.line_chart(history_chart_df.set_index("date")[["close"]])
        else:
            st.info("No historical data available for this selection.")

        st.subheader("Volume")
        if not history_chart_df.empty:
            st.bar_chart(history_chart_df.set_index("date")[["volume"]])
        else:
            st.info("No volume data available for this selection.")

    with hero_right:
        st.subheader("Next-Day Prediction")
        st.metric("Latest Close", f"{prediction_payload['latest_close']:.2f}")
        delta = prediction_payload["next_day_prediction"] - prediction_payload["latest_close"]
        st.metric("Predicted Close", f"{prediction_payload['next_day_prediction']:.2f}", f"{delta:.2f}")
        st.write(f"Prediction source: `{prediction_payload['prediction_source']}`")

        st.subheader("Baseline Predictions")
        baseline_prediction_df = build_baseline_prediction_table(prediction_payload["baselines"])
        st.dataframe(baseline_prediction_df, use_container_width=True, hide_index=True)

    st.subheader("Forecast")
    if forecast_df.empty:
        st.info("Forecast output is not available yet.")
    else:
        st.line_chart(forecast_df.set_index("step")[["predicted_close"]])
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    st.subheader("Actual vs Predicted")
    if comparison_df.empty:
        st.info("Saved comparison data is not available yet.")
    else:
        st.line_chart(comparison_df.set_index("index")[["actual", "predicted"]])

    st.subheader("LSTM Metrics")
    lstm_metrics = metrics_payload.get("lstm", {}).get(comparison_split, {})
    metric_card_columns(lstm_metrics)

    st.subheader(f"{comparison_split.title()} Baseline Metrics")
    baseline_metric_df = build_baseline_metric_table(metrics_payload, comparison_split)
    if baseline_metric_df.empty:
        st.info("No baseline metric summary is available yet.")
    else:
        st.dataframe(baseline_metric_df, use_container_width=True, hide_index=True)

    st.subheader("Explainability")
    explain_left, explain_right = st.columns(2)
    with explain_left:
        st.write("Model target")
        st.code(model_info["target_col"])
        st.write("Features used")
        st.code(", ".join(model_info["feature_cols"]))
    with explain_right:
        st.write("How to read this dashboard")
        st.markdown(
            """
            - `Lookback days` controls how many recent rows are fed into the model.
            - `Forecast horizon` uses repeated next-step inference for short-range projection.
            - `Prediction mode` tells you whether the trained LSTM is active or the app fell back to a baseline.
            - `Comparison split` switches the saved evaluation chart between validation and test data.
            """
        )

    st.subheader("Limitations")
    st.markdown(
        """
        - Forecast quality depends on whether saved training artifacts match the current engineered feature set.
        - Multi-step forecasts are recursive, so error can compound across forecast steps.
        - Historical metrics reflect the saved evaluation run, not live market validation.
        - This dashboard is for experimentation and education, not trading decisions.
        """
    )

    st.warning(model_info["disclaimer"])


if __name__ == "__main__":
    main()
