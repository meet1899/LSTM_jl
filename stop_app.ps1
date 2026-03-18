$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = (Join-Path $projectRoot ".venv\Scripts\python.exe").ToLower()

function Stop-ProjectProcess {
    param(
        [string]$Pattern,
        [string]$Label
    )

    $candidates = Get-CimInstance Win32_Process | Where-Object {
        $_.ExecutablePath -and
        $_.ExecutablePath.ToLower() -eq $pythonExe -and
        $_.CommandLine -and
        $_.CommandLine -like "*$Pattern*"
    }

    if (-not $candidates) {
        Write-Host "$Label is not running."
        return
    }

    foreach ($process in $candidates) {
        Stop-Process -Id $process.ProcessId -Force
        Write-Host "Stopped $Label (PID $($process.ProcessId))."
    }
}

Stop-ProjectProcess -Pattern "uvicorn app.backend.main:app" -Label "FastAPI backend"
Stop-ProjectProcess -Pattern "streamlit run app/frontend/streamli_app.py" -Label "Streamlit app"
