@echo off
REM Compare Transcription Variants
REM Runs variants 7, 17, 18, 19 (only enabled variants)

setlocal enabledelayedexpansion

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"

REM Convert Windows path to WSL path
set "WSL_INPUT_DIR=!SCRIPT_DIR:\=/!"
if "!SCRIPT_DIR:~0,1!"=="C" (
    set "WSL_INPUT_DIR=!WSL_INPUT_DIR:C:=/mnt/c!"
)
if "!SCRIPT_DIR:~0,1!"=="D" (
    set "WSL_INPUT_DIR=!WSL_INPUT_DIR:D:=/mnt/d!"
)

echo ========================================
echo Running Transcription Comparison Script
echo Using NEW Variant System
echo ========================================
echo.
echo Processing audio files in: %SCRIPT_DIR%
echo WSL path: !WSL_INPUT_DIR!
echo.

REM Find the audio file in the directory
for %%f in ("!SCRIPT_DIR!\*.wav" "!SCRIPT_DIR!\*.mp3" "!SCRIPT_DIR!\*.m4a") do (
    set "AUDIO_FILE=%%f"
    goto :found
)

echo No audio file found in directory!
pause
exit /b 1

:found
echo Audio file: !AUDIO_FILE!
echo.

REM Convert audio file path to WSL format
set "WSL_AUDIO_FILE=!AUDIO_FILE:\=/!"
if "!AUDIO_FILE:~0,1!"=="C" (
    set "WSL_AUDIO_FILE=!WSL_AUDIO_FILE:C:=/mnt/c!"
)
if "!AUDIO_FILE:~0,1!"=="D" (
    set "WSL_AUDIO_FILE=!WSL_AUDIO_FILE:D:=/mnt/d!"
)

REM Run the comparison script via WSL
REM This will automatically use only enabled variants
wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && cd /home/kristjans/projects/stt-faster && .venv/bin/python scripts/compare_transcription_variants.py '!WSL_AUDIO_FILE!' --preset et-large --language et"

echo.
echo ========================================
echo Processing Complete!
echo ========================================

pause