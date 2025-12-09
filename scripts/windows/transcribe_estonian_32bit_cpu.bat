@echo off
REM Audio Transcription - ESTONIAN Language (32-bit CPU, Variant 7)
REM Uses: TalTech Estonian Whisper model (et-32 preset, CPU-optimized)
REM Variant 7: No preprocessing + minimal transcription parameters (no VAD filter)
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
echo Audio Transcription - ESTONIAN (Variant 7)
echo ========================================
echo.
echo Model: TalTech Estonian Whisper (et-32, CPU-optimized)
echo Language: Estonian
echo Variant: 7 (No preprocessing + minimal params, no VAD filter)
echo Processing audio files in: %SCRIPT_DIR%
echo WSL path: !WSL_INPUT_DIR!
echo.

REM Run the transcription script via WSL with variant 7
REM Environment variables:
REM   HF_HOME - Hugging Face cache location
REM   HF_HUB_CACHE - Hugging Face hub cache
REM Output: Both txt and json formats
REM Note: Using delayed expansion variable with single quotes in bash command
wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && cd /home/kristjans/projects/stt-faster && .venv/bin/python scripts/transcribe_manager.py process '!WSL_INPUT_DIR!' --preset et-32 --language et --output-format both --variant 7"

echo.
echo ========================================
echo Processing Complete!
echo ========================================
echo.
echo Processed files are in: %~dp0processed\
echo Failed files are in: %~dp0failed\
echo.

pause

