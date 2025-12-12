@echo off
REM Audio Transcription - Custom Variant List (NEW Implementation)
REM Uses: New multi-variant runner with --variants option
REM Allows you to specify custom variant numbers
REM Processes a single audio file through specified variants
REM Outputs are organized in variant-specific subfolders

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
if "!SCRIPT_DIR:~0,1!"=="E" (
    set "WSL_INPUT_DIR=!WSL_INPUT_DIR:E:=/mnt/e!"
)

REM Find the audio file in the directory
for %%f in ("!SCRIPT_DIR!\*.wav" "!SCRIPT_DIR!\*.mp3" "!SCRIPT_DIR!\*.m4a" "!SCRIPT_DIR!\*.flac" "!SCRIPT_DIR!\*.ogg") do (
    set "AUDIO_FILE=%%f"
    goto :found
)

echo No audio file found in directory!
echo Supported formats: .wav, .mp3, .m4a, .flac, .ogg
pause
exit /b 1

:found
echo Audio file: !AUDIO_FILE!
echo.

REM Prompt for variant list (default: conservative sweep)
set /p VARIANTS="Enter variant numbers (comma-separated, e.g., 1,6,21,22) or press Enter for conservative sweep (1,6,21,22,23,24,25,26): "
if "!VARIANTS!"=="" (
    set "VARIANTS=1,6,21,22,23,24,25,26"
    echo Using conservative sweep: !VARIANTS!
) else (
    echo Using custom variants: !VARIANTS!
)

echo.
echo ========================================
echo Custom Variant Comparison (NEW)
echo ========================================
echo.
echo Model: TalTech Estonian Whisper (et-large)
echo Language: Estonian
echo Variants: !VARIANTS!
echo Processing audio file: !AUDIO_FILE!
echo WSL path: !WSL_INPUT_DIR!
echo.

REM Run the transcription script via WSL with custom variant list
REM Environment variables:
REM   HF_HOME - Hugging Face cache location
REM   HF_HUB_CACHE - Hugging Face hub cache
REM Output: Both txt and json formats
REM Output structure: variant_outputs/variant_XXX_name/processed/
REM Note: transcribe_manager.py processes folders, so we pass the directory containing the file
wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && cd /home/kristjans/projects/stt-faster && .venv/bin/python scripts/transcribe_manager.py process '!WSL_INPUT_DIR!' --preset et-large --language et --output-format both --variants '!VARIANTS!'"

echo.
echo ========================================
echo Processing Complete!
echo ========================================
echo.
echo Outputs are organized in: %~dp0variant_outputs\
echo Each variant has its own folder with processed/ and run_meta.json
echo.

pause

