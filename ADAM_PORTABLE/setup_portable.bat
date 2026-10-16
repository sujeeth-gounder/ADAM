@echo off
REM ==============================================================================
REM ADAM PORTABLE — Windows Setup Script
REM ==============================================================================

cd /d "%~dp0"
title ADAM Portable — Windows Setup

echo ==============================================================================
echo           A D A M  PORTABLE  —  Windows 10 / 11 Setup Wizard
echo ==============================================================================

REM 1. Create Virtualenv
echo.
echo [1/4] Configuring Python Virtual Environment...
if not exist "venv" (
    python -m venv venv
    echo   Created venv in .\venv
) else (
    echo   Found existing venv.
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

REM 2. Install Requirements
echo.
echo [2/4] Installing Python Requirements...
pip install -r requirements.txt

REM 3. Download Piper Amy Voice
echo.
echo [3/4] Downloading Piper Amy English Voice Models...
python scripts\download_models.py

REM 4. Check Ollama
echo.
echo [4/4] Checking Ollama Status on Windows...
where ollama >nul 2>nul
if %errorlevel% equ 0 (
    echo   Ollama CLI detected.
    echo   Pulling model 'qwen2.5:1.5b'...
    ollama pull qwen2.5:1.5b
) else (
    echo   Ollama is not installed on this PC.
    echo   Download the Windows installer from: https://ollama.com/download/windows
)

echo.
echo ==============================================================================
echo           ADAM Portable is ready to launch on Windows!
echo ==============================================================================
echo To start ADAM:
echo     run_adam.bat
echo.
pause
