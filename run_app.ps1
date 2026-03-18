$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment python not found at $pythonExe"
}

Write-Host "Starting FastAPI backend on http://127.0.0.1:8000 ..."
Start-Process -FilePath $pythonExe `
    -ArgumentList @("-m", "uvicorn", "app.backend.main:app", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory $projectRoot

Write-Host "Starting Streamlit app on http://127.0.0.1:8501 ..."
Start-Process -FilePath $pythonExe `
    -ArgumentList @("-m", "streamlit", "run", "app/frontend/streamli_app.py", "--server.headless", "true", "--server.port", "8501") `
    -WorkingDirectory $projectRoot

Write-Host ""
Write-Host "App started."
Write-Host "FastAPI docs: http://127.0.0.1:8000/docs"
Write-Host "Streamlit app: http://127.0.0.1:8501"
Write-Host ""
Write-Host "If the browser does not open automatically, paste those URLs into your browser."
