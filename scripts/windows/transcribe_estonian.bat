@echo off
REM Audio Transcription - ESTONIAN Language
REM Uses: TalTech Estonian Whisper model (et-large preset)
REM Note: Uses HF cache at ~/.cache/hf/ for faster model loading
REM
REM ========================================
REM CONFIGURATION: Edit variant numbers here
REM ========================================
REM Specify variant numbers separated by spaces (e.g., "1 36 44" or "44")
REM For single variant, just use one number (e.g., "16")
set "VARIANTS=16"

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

REM Convert space-separated variant numbers to comma-separated list
REM Count variants and build comma-separated string
set "VARIANTS_COMMA="
set /a VARIANT_COUNT=0
for %%v in (!VARIANTS!) do (
    if "!VARIANTS_COMMA!"=="" (
        set "VARIANTS_COMMA=%%v"
    ) else (
        set "VARIANTS_COMMA=!VARIANTS_COMMA!,%%v"
    )
    set /a VARIANT_COUNT+=1
)
set "VARIANTS=!VARIANTS_COMMA!"

echo ========================================
echo Audio Transcription - ESTONIAN
echo ========================================
echo.
echo Model: TalTech Estonian Whisper (large-v3-turbo)
echo Language: Estonian
if !VARIANT_COUNT!==1 (
    echo Variant: !VARIANTS!
) else (
    echo Variants: !VARIANTS! ^(!VARIANT_COUNT! variants^)
)
echo Processing audio files in: %SCRIPT_DIR%
echo WSL path: !WSL_INPUT_DIR!
echo.

REM Run the transcription script via WSL
REM Environment variables:
REM   HF_HOME - Hugging Face cache location
REM   HF_HUB_CACHE - Hugging Face hub cache
REM Output: Both txt and json formats
REM Note: Using delayed expansion variable with single quotes in bash command
REM Use --variant for single variant, --variants for multiple variants
if !VARIANT_COUNT!==1 (
    REM Single variant - use --variant for backward compatibility
    wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && cd /home/kristjans/projects/stt-faster && .venv/bin/python scripts/transcribe_manager.py process '!WSL_INPUT_DIR!' --language et --output-format both --variant !VARIANTS!"
) else (
    REM Multiple variants - use --variants with comma-separated list
    wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && cd /home/kristjans/projects/stt-faster && .venv/bin/python scripts/transcribe_manager.py process '!WSL_INPUT_DIR!' --language et --output-format both --variants '!VARIANTS!'"
)

echo.
echo ========================================
echo Processing Complete!
echo ========================================
echo.
echo Processed files are in: %~dp0processed\
echo Failed files are in: %~dp0failed\
echo.

pause
