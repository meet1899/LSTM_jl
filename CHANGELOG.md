# Changelog

## Current State

- Refactored notebook workflow into modular `src/`, backend, and frontend layers
- Implemented leakage-safe preprocessing and split logic
- Added evaluation metrics and baseline models
- Added engineered features
- Added configurable training, checkpointing, and artifact saving
- Built FastAPI backend and Streamlit frontend
- Added local ingestion validation and refresh metadata
- Added logging and request timing
- Added a passing pytest suite

## Major Milestones

### Step 3A-3D

- removed preprocessing leakage
- added train/validation/test logic
- added evaluation metrics
- added baseline comparisons

### Step 4-6

- added engineered features
- improved training with callbacks and config-driven hyperparameters
- saved artifacts to `models/` and `results/`

### Step 8-10

- built backend API routes
- built Streamlit dashboard
- added run/stop scripts
- added validated local data ingestion and refresh metadata

### Step 11-12

- added multi-layer test suite
- added centralized logging and backend request timing
