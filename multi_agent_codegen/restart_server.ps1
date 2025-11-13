# PowerShell script to restart the Flask server

Write-Host "Stopping any running Flask servers..." -ForegroundColor Yellow

# Try to find and stop Flask processes on port 5000
$port = 5000
$processes = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique

if ($processes) {
    foreach ($pid in $processes) {
        try {
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            Write-Host "Stopped process on port $port (PID: $pid)" -ForegroundColor Green
        } catch {
            Write-Host "Could not stop process $pid" -ForegroundColor Red
        }
    }
} else {
    Write-Host "No process found on port $port" -ForegroundColor Cyan
}

Start-Sleep -Seconds 2

Write-Host "`nStarting Flask server..." -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Cyan

# Start the server
python run.py

