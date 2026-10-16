@echo off
REM ==============================================================================
REM ADAM PORTABLE — Windows 10 / 11 Launcher
REM ==============================================================================

cd /d "%~dp0"
title ADAM — Advanced Desktop AI Mate (Portable)

echo ==============================================================================
echo       A D A M  —  Advanced Desktop AI Mate (Windows Portable)
echo ==============================================================================

REM 1. Activate Virtual Environment if available
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    if exist "..\venv\Scripts\activate.bat" (
        call ..\venv\Scripts\activate.bat
    ) else (
        echo [INFO] No local venv detected. Running with system Python...
    )
)

REM 2. Check and start Ollama on Windows
where ollama >nul 2>nul
if %errorlevel% equ 0 (
    powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -TimeoutSec 1; exit 0 } catch { exit 1 }" >nul 2>nul
    if %errorlevel% neq 0 (
        echo Starting Ollama background server...
        start /b ollama serve >nul 2>nul
        timeout /t 2 >nul
    )
)

REM 3. Set PythonPath and Launch
set PYTHONPATH=%~dp0;%PYTHONPATH%
python run_adam.py %*

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] ADAM exited with code %errorlevel%.
    pause
)
