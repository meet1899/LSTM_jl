# LSTM Stock Predictor

End-to-end time-series forecasting app for stock price prediction with a modular ML pipeline, baseline comparison, FastAPI backend, and Streamlit dashboard.

## Overview

This project started as a notebook-based stock prediction experiment and has been refactored into a structured application with:

- a training pipeline in `src/`
- a FastAPI backend in `app/backend`
- a Streamlit frontend in `app/frontend`
- local CSV-based ingestion and refresh metadata
- artifact saving for models, scalers, metrics, and predictions
- a pytest suite covering the ML core, ingestion layer, and backend API

The current local dataset is Microsoft (`MSFT`) historical stock data from `data/MicrosoftStock.csv`.

## Current Status

Implemented so far:

- leakage-safe chronological train/validation/test split
- sequence generation for time-series windows
- engineered features beyond `close`
- LSTM training with early stopping and checkpointing
- baseline comparisons:
  - naive last value
  - moving average
  - linear regression
- artifact saving to `models/` and `results/`
- FastAPI endpoints for health, model info, history, prediction, forecast, metrics, and comparison
- Streamlit dashboard for Version 1 feature set
- local data ingestion validation and refresh metadata
- logging and request timing
- test suite with `26` passing tests

Known current limitation:

- the backend is currently falling back to `naive_last_value_fallback` because the saved `.keras` artifact is not loadable against the current feature-engineered pipeline state

## Project Structure

```text
app/
  backend/
    main.py
    schemas.py
    services.py
  frontend/
    streamli_app.py
data/
  MicrosoftStock.csv
  refresh_metadata.json
models/
results/
notebooks/
src/
  baselines.py
  config.py
  data_loader.py
  evaluate.py
  features.py
  logging_config.py
  preprocessing.py
  sequence.py
  train.py
  update_data.py
  utils.py
tests/
run_app.ps1
stop_app.ps1
plan.txt
executed.txt
```

## Local Setup

Use the project virtual environment.

### 1. Activate `.venv`

```powershell
.\.venv\Scripts\Activate.ps1
```

Or call the interpreter explicitly:

```powershell
.\.venv\Scripts\python.exe --version
```

### 2. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-backend.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-frontend.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Running the App

### Option 1: use the launcher script

```powershell
.\run_app.ps1
```

This starts:

- FastAPI on `http://127.0.0.1:8000`
- Streamlit on `http://127.0.0.1:8501`

Useful URLs:

- FastAPI docs: `http://127.0.0.1:8000/docs`
- Streamlit app: `http://127.0.0.1:8501`

To stop both:

```powershell
.\stop_app.ps1
```

### Option 2: run services manually

Backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/frontend/streamli_app.py --server.headless true --server.port 8501
```

## Docker Deployment

This repo now supports separate backend and frontend container images.

### Files

- [Dockerfile.backend](/C:/Users/meetp/files/PROJECTS/LSTM_jl/Dockerfile.backend)
- [Dockerfile.frontend](/C:/Users/meetp/files/PROJECTS/LSTM_jl/Dockerfile.frontend)
- [compose.yaml](/C:/Users/meetp/files/PROJECTS/LSTM_jl/compose.yaml)
- [.dockerignore](/C:/Users/meetp/files/PROJECTS/LSTM_jl/.dockerignore)

### Build and run locally with Docker Compose

```powershell
docker compose up --build
```

This starts:

- backend on `http://127.0.0.1:8000`
- frontend on `http://127.0.0.1:8501`

### Stop Docker services

```powershell
docker compose down
```

### Deployment notes

- the frontend reads the backend URL from `API_BASE_URL`
- in Compose, it is set to `http://backend:8000`
- backend and frontend are built as separate services/images
- both images include container healthchecks
- both images run as a non-root user

### Split requirements

- [requirements-backend.txt](/C:/Users/meetp/files/PROJECTS/LSTM_jl/requirements-backend.txt): backend runtime dependencies
- [requirements-frontend.txt](/C:/Users/meetp/files/PROJECTS/LSTM_jl/requirements-frontend.txt): frontend runtime dependencies
- [requirements-dev.txt](/C:/Users/meetp/files/PROJECTS/LSTM_jl/requirements-dev.txt): development and test dependencies

## Training Pipeline

The current pipeline performs:

1. local dataset loading and validation
2. feature engineering
3. chronological split
4. train-only scaling
5. sequence creation
6. LSTM training with validation monitoring
7. baseline comparison
8. artifact saving

Run the metadata refresh:

```powershell
.\.venv\Scripts\python.exe src\update_data.py
```

Training currently runs through `src/train.py`. The code supports:

- saved model artifacts
- saved scalers
- saved metrics and prediction outputs
- checkpointing with `models/best_lstm.keras`

## Evaluation Methodology

The project uses:

- chronological `train / validation / test` splitting
- train-only scaling to avoid leakage
- baseline comparisons against simple forecasting methods
- metrics:
  - MAE
  - RMSE
  - MAPE
  - direction accuracy

This makes model quality easier to judge honestly instead of evaluating the LSTM in isolation.

## Backend API

Current routes:

- `GET /health`
- `GET /model-info`
- `GET /history`
- `POST /predict`
- `POST /forecast`
- `GET /metrics`
- `GET /comparison`

### Backend notes

- `/predict` and `/forecast` will use the trained LSTM if artifacts are valid
- otherwise they fall back to naive last-value predictions
- `/model-info` exposes:
  - available baselines
  - whether artifacts are ready
  - prediction mode
  - data refresh metadata
  - model metadata when available

## Frontend Dashboard

The Streamlit dashboard currently includes:

- ticker selection
- lookback control
- forecast horizon control
- historical price chart
- volume chart
- next-day prediction
- baseline prediction comparison
- forecast display
- actual vs predicted chart
- LSTM metric cards
- baseline metric table
- explainability and limitations sections

## Testing

Run the full test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Current result:

- `26 passed`

Test coverage includes:

- preprocessing
- sequence generation
- feature engineering
- ingestion
- evaluation
- baselines
- training helpers
- backend API contracts

## Logging

The project now includes centralized logging for:

- backend request timing
- backend success/failure events
- ingestion refresh operations
- training lifecycle events
- artifact load failures and fallback prediction behavior

## Limitations

- The current LSTM underperforms simple baselines on the local dataset.
- Prediction currently falls back when the saved `.keras` artifact cannot be loaded.
- Data ingestion is currently local CSV-based; live ingestion is not implemented yet.
- Full default training can be slow depending on environment/runtime limits.
- The app is for experimentation and education, not production trading use.

## Disclaimer

For educational use only. This project does not provide financial advice.

## Screenshots

Suggested screenshots to add later:

- FastAPI docs page
- Streamlit dashboard home view
- forecast section
- actual vs predicted chart

## Related Project Files

- Plan: [plan.txt](/C:/Users/meetp/files/PROJECTS/LSTM_jl/plan.txt)
- Implementation log: [executed.txt](/C:/Users/meetp/files/PROJECTS/LSTM_jl/executed.txt)
