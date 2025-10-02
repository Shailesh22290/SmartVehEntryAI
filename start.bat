@echo off
echo ========================================
echo SmartVehEntryAI - Starting Services
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Starting Staff Interface on port 8000...
start "PlateVision - Staff" cmd /k "uvicorn app:app --reload --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak > nul

echo Starting Admin Panel on port 8001...
start "PlateVision - Admin" cmd /k "uvicorn admin:admin --reload --host 0.0.0.0 --port 8001"

echo.
echo ========================================
echo Services Started!
echo ========================================
echo Staff Interface: http://localhost:8000
echo Admin Panel:     http://localhost:8001
echo.
echo Press any key to stop all services...
pause > nul

taskkill /FI "WINDOWTITLE eq PlateVision - Staff*" /T /F
taskkill /FI "WINDOWTITLE eq PlateVision - Admin*" /T /F

echo Services stopped.
pause