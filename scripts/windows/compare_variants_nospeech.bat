@echo off
REM Audio Transcription - No-Speech Threshold Validation (NEW Implementation)
REM Uses: New multi-variant runner with --variants option
REM Processes a single audio file through variants 1,90,91,92,93,94,95,96
REM Variants 90-93: Validation sweep (diagnostic, vad_filter=False)
REM Variants 94-96: Production candidates (vad_filter=True)
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

REM Set variant list: baseline + validation sweep + production candidates
set "VARIANTS=1,90,91,92,93,94,95,96"

echo ========================================
echo No-Speech Threshold Validation
echo ========================================
echo.
echo Model: TalTech Estonian Whisper (et-large)
echo Language: Estonian
echo Variants: !VARIANTS!
echo Processing audio file: !AUDIO_FILE!
echo WSL path: !WSL_INPUT_DIR!
echo.
echo Variant details:
echo   - Variant 1: Baseline (noprep_noparamtrans)
echo   - Variant 90: Diagnostic baseline (vad_filter=False)
echo   - Variant 91: Diagnostic conservative (ns=0.75, lp=-0.9, vad_filter=False)
echo   - Variant 92: Diagnostic balanced (ns=0.65, lp=-0.8, vad_filter=False)
echo   - Variant 93: Diagnostic active (ns=0.60, lp=-0.7, vad_filter=False)
echo   - Variant 94: Production safest (ns=0.75, lp=-0.8)
echo   - Variant 95: Production balanced (ns=0.70, lp=-0.7)
echo   - Variant 96: Production aggressive (ns=0.65, lp=-0.7)
echo   (ns=no_speech_threshold, lp=logprob_threshold)
echo.

REM Run the transcription script via WSL with variant list
REM Environment variables:
REM   HF_HOME - Hugging Face cache location
REM   HF_HUB_CACHE - Hugging Face hub cache
REM Output: Both txt and json formats
REM Output structure: timestamped folder (e.g., 2024-01-15T10-30-45) directly under input directory
REM Note: transcribe_manager.py processes folders, so we pass the directory containing the file
wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && cd /home/kristjans/projects/stt-faster && .venv/bin/python scripts/transcribe_manager.py process '!WSL_INPUT_DIR!' --preset et-large --language et --output-format both --variants '!VARIANTS!'"

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
echo To generate a comparison report, run:
echo   wsl -e bash -c "cd /home/kristjans/projects/stt-faster && .venv/bin/python scripts/generate_variant_report.py '!WSL_INPUT_DIR!' --variants 1,90,91,92,93,94,95,96 --far-speaker-range 20-60 --silence-range 120-160"
echo.
echo (Adjust time ranges as needed for your audio file)
echo.

pause

