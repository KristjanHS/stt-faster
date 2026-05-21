@echo off
REM stt-faster - native Windows installer.
REM Creates a fresh .venv, installs the package editable, optionally prefetches
REM Whisper models. WSL is never bootstrapped here; the transcribe bats decide
REM at runtime whether to use WSL or this .venv (see scripts\windows\_runtime.bat).

git pull
if errorlevel 1 (
    echo WARNING: git pull failed. Continuing with the local checkout.
)

echo === stt-faster - Setup ===
echo.

py -3.12 --version >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.12 not found via the py launcher.
    echo Install Python 3.12 from https://www.python.org/downloads/
    echo During install, check "Add to PATH" and "Install launcher".
    pause
    exit /b 1
)

if exist .venv (
    echo Removing stale .venv...
    rmdir /s /q .venv
    if errorlevel 1 (
        echo ERROR: Failed to remove existing .venv directory. Close any process using it and retry.
        pause
        exit /b 1
    )
)

echo Creating virtual environment with Python 3.12...
py -3.12 -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

echo Installing dependencies (pip install -e .)...
call .venv\Scripts\activate.bat
pip install -e .
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

REM ---- Torch guard ---------------------------------------------------------
REM faster-whisper only pulls torch via its [conversion] extra; ctranslate2 is
REM the runtime path. If torch shows up here, a transitive dep has changed.
pip list 2>nul | findstr /I /R /C:"^torch " >nul
if not errorlevel 1 (
    echo.
    echo !!! WARNING: torch was installed by a transitive dependency.
    echo !!! Expected: torch absent (ctranslate2 is the runtime, not torch).
    echo !!! Setup will continue, but check pyproject.toml for new dep drift.
    echo.
)

REM ---- Model prefetch prompt -----------------------------------------------
echo.
set /p PREFETCH=Prefetch Whisper models for Estonian and English now? (~3 GB) [y/N]:
if /i "%PREFETCH%"=="y" (
    set "HF_HOME=%USERPROFILE%\.cache\hf"
    set "HF_HUB_CACHE=%USERPROFILE%\.cache\hf\hub"
    .venv\Scripts\python scripts\prefetch_models.py
    if errorlevel 1 (
        echo WARNING: Prefetch failed. The first transcription run will download lazily instead.
    )
) else (
    echo Skipping prefetch; models will download on first transcription run.
)

echo.
echo === Setup complete ===
echo.
echo Next steps:
echo   1. (optional) set STT_AUDIO_DIR=C:\path\to\audio    Persist with setx if you want it across sessions.
echo   2. Run a transcribe bat, either:
echo      - Copy scripts\windows\transcribe_*.bat into the audio folder and double-click, OR
echo      - With STT_AUDIO_DIR set, run scripts\windows\transcribe_*.bat in place.
echo.
echo Notes:
echo   - Native Windows runs CPU-only unless CUDA + cuDNN are installed manually.
echo     The bats print a highly-visible banner when GPU fallback fires.
echo   - The bats auto-detect WSL: if WSL has the legacy repo + .venv, they use it;
echo     otherwise they use this Windows .venv.
echo.
pause
