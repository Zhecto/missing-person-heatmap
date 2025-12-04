# START-SERVER.bat
# Windows batch script to start the backend server

@echo off
echo =========================================
echo  Missing Person Heatmap Analysis Server
echo =========================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)
echo.

echo Starting FastAPI backend server...
echo.
echo Server will be available at:
echo   - Web Interface: http://localhost:8000
echo   - API Docs: http://localhost:8000/docs
echo.
echo Press CTRL+C to stop the server
echo.

cd /d "%~dp0"
uvicorn src.backend.main:app --reload --host 0.0.0.0 --port 8000

pause
