@echo off
REM Audio Transcription - ENGLISH/Multilingual
REM Uses: Whisper large-v3 model (large8gb preset)
REM Note: Uses HF cache at ~/.cache/hf/ for faster model loading

setlocal enabledelayedexpansion

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"

REM Convert Windows path to WSL path
REM First, replace backslashes with forward slashes
set "WSL_INPUT_DIR=!SCRIPT_DIR:\=/!"
REM Then replace drive letter with /mnt/drive format
if "!SCRIPT_DIR:~0,1!"=="C" (
    set "WSL_INPUT_DIR=!WSL_INPUT_DIR:C:=/mnt/c!"
)
if "!SCRIPT_DIR:~0,1!"=="D" (
    set "WSL_INPUT_DIR=!WSL_INPUT_DIR:D:=/mnt/d!"
)
if "!SCRIPT_DIR:~0,1!"=="E" (
    set "WSL_INPUT_DIR=!WSL_INPUT_DIR:E:=/mnt/e!"
)
REM Ensure path starts with /mnt
if not "!WSL_INPUT_DIR:~0,4!"=="/mnt" (
    echo Error: Could not convert path to WSL format
    echo Original path: !SCRIPT_DIR!
    pause
    exit /b 1
)

echo ========================================
echo Audio Transcription - ENGLISH
echo ========================================
echo.
echo Model: Whisper Turbo (distil-large-v3)
echo Language: English / Multilingual
echo Pre-processing: enabled (downmix/resample + loudnorm)
echo Processing audio files in: %SCRIPT_DIR%
echo WSL path: !WSL_INPUT_DIR!
echo.

REM Run the transcription script via WSL with turbo preset
REM Set HF_HOME to use the correct cache location
REM Force English language to avoid auto-detection errors
REM Enable light audio pre-processing (downmix/resample + loudnorm)
REM Output both txt and json formats
REM Note: Using delayed expansion variable with single quotes in bash command
wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && export STT_PREPROCESS_ENABLED=1 && cd /home/kristjans/projects/stt-faster && .venv/bin/python scripts/transcribe_manager.py process '!WSL_INPUT_DIR!' --preset turbo --language en --output-format both"

echo.
echo ========================================
echo Processing Complete!
echo ========================================
echo.
echo Processed files are in: %~dp0processed\
echo Failed files are in: %~dp0failed\
echo.

pause
