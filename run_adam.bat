@echo off
REM ==============================================================================
REM ADAM — Advanced Desktop AI Mate (Windows Launcher)
REM ==============================================================================

cd /d "%~dp0"

REM 1. Activate venv if exists
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

REM 2. Check and start Ollama in background if installed
where ollama >nul 2>nul
if %ERRORLEVEL% equ 0 (
    powershell -NoProfile -Command "try { $res = Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -TimeoutSec 1; } catch { Start-Process ollama -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep -Seconds 1 }" >nul 2>nul
)

REM 3. Launch ADAM
set PYTHONPATH=%~dp0;%PYTHONPATH%
python run_adam.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ADAM] Application exited with error code %ERRORLEVEL%.
    pause
)
