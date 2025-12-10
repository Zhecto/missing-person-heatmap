# PowerShell script to start the backend server

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " Missing Person Heatmap Analysis Server" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# Check if dependencies are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$packages = @("fastapi", "uvicorn", "pandas", "scikit-learn")
$missingPackages = @()

foreach ($package in $packages) {
    $installed = pip show $package 2>&1 | Select-String "Name:"
    if ($installed) {
        Write-Host "  ✓ $package" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $package (missing)" -ForegroundColor Red
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host ""
    Write-Host "Missing packages detected!" -ForegroundColor Yellow
    $install = Read-Host "Install missing packages now? (y/n)"
    if ($install -eq "y") {
        Write-Host "Installing dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
    } else {
        Write-Host "Cannot start server without dependencies." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}
Write-Host ""

# Start the server
Write-Host "Starting FastAPI backend server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Server will be available at:" -ForegroundColor Cyan
Write-Host "  - Web Interface: http://localhost:8000" -ForegroundColor White
Write-Host "  - API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "  - Health Check: http://localhost:8000/health" -ForegroundColor White
Write-Host ""
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Change to script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Start uvicorn
uvicorn src.backend.main:app --reload --host 0.0.0.0 --port 8000
