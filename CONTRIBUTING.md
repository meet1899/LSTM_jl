# Contributing

## Environment

Use the project virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or call it explicitly:

```powershell
.\.venv\Scripts\python.exe
```

## Run the app

```powershell
.\run_app.ps1
```

Stop it with:

```powershell
.\stop_app.ps1
```

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Data refresh

```powershell
.\.venv\Scripts\python.exe src\update_data.py
```

## General guidelines

- keep new code inside the existing `src/`, `app/backend`, `app/frontend`, and `tests/` structure
- add tests for new ML, ingestion, or API behavior
- prefer using the `.venv` interpreter explicitly in scripts and commands
- avoid reintroducing data leakage into preprocessing or evaluation
- keep fallback behavior explicit and logged
