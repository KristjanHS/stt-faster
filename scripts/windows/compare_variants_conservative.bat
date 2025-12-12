@echo off
REM Audio Transcription - Conservative Variant Sweep (NEW Implementation)
REM Uses: New multi-variant runner with --variant-sweep conservative
REM Processes a single audio file through variants 1,6,21,22,23,24,25,26
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

echo ========================================
echo Conservative Variant Sweep (NEW)
echo ========================================
echo.
echo Model: TalTech Estonian Whisper (et-large)
echo Language: Estonian
echo Variants: 1,6,21,22,23,24,25,26 (conservative sweep)
echo Processing audio files in: %SCRIPT_DIR%
echo WSL path: !WSL_INPUT_DIR!
echo.

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
echo Variant details:
echo   - Variant 1: Baseline (noprep_noparamtrans)
echo   - Variant 6: condition_on_previous_text=False
echo   - Variant 21: beam_size=4
echo   - Variant 22: beam_size=6
echo   - Variant 23: chunk_length=18
echo   - Variant 24: chunk_length=22
echo   - Variant 25: no_speech_threshold=0.45
echo   - Variant 26: no_speech_threshold=0.55
echo.

REM Run the transcription script via WSL with conservative sweep
REM Environment variables:
REM   HF_HOME - Hugging Face cache location
REM   HF_HUB_CACHE - Hugging Face hub cache
REM Output: Both txt and json formats
REM Output structure: timestamped folder (e.g., 2024-01-15T10-30-45) directly under input directory
REM Note: transcribe_manager.py processes folders, so we pass the directory containing the file
wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && cd /home/kristjans/projects/stt-faster && .venv/bin/python scripts/transcribe_manager.py process '!WSL_INPUT_DIR!' --preset et-large --language et --output-format both --variant-sweep conservative"

echo.
echo ========================================
echo Processing Complete!
echo ========================================
echo.
echo Outputs are organized in timestamped folders directly under: %~dp0
echo All variants are in a single timestamped folder (e.g., 2024-01-15T10-30-45)
echo Files are prefixed with variant number (e.g., variant_001_audio.wav.txt)
echo A single run_meta.json contains metadata for all variants
echo.
echo   - Variant 1: Baseline (noprep_noparamtrans)
echo   - Variant 6: condition_on_previous_text=False
echo   - Variant 21: beam_size=4
echo   - Variant 22: beam_size=6
echo   - Variant 23: chunk_length=18
echo   - Variant 24: chunk_length=22
echo   - Variant 25: no_speech_threshold=0.45
echo   - Variant 26: no_speech_threshold=0.55
echo.

pause

