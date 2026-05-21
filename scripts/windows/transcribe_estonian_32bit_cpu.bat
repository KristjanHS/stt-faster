@echo off
REM Audio Transcription - ESTONIAN, CPU-only (et-32 preset, variant 7)
REM Variant 7: no preprocessing + minimal transcription parameters (no VAD filter).
REM Edit the VARIANTS line below to use a different variant.
set "VARIANTS=7"

setlocal enabledelayedexpansion
set "STT_CALLER_DIR=%~dp0"
call "%~dp0_runtime.bat"
if errorlevel 1 (
    pause
    exit /b 1
)

REM Normalize space-separated VARIANTS to comma-separated for --variants.
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

echo ========================================
echo Audio Transcription - ESTONIAN (32-bit CPU)
echo ========================================
echo Model:    TalTech Estonian Whisper (et-32, CPU-optimized)
echo Language: Estonian
if !VARIANT_COUNT!==1 (echo Variant:  !VARIANTS_COMMA!) else (echo Variants: !VARIANTS_COMMA! ^(!VARIANT_COUNT! variants^))
echo.

if /i "!STT_RUNTIME!"=="wsl" (
    wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && cd !STT_WSL_REPO! && .venv/bin/python scripts/transcribe_manager.py process '!STT_AUDIO_DIR_WSL!' --preset et-32 --language et --output-format txt --diarize --num-speakers 2 --variants '!VARIANTS_COMMA!'"
) else (
    docker run --rm ^
      -v "!STT_AUDIO_DIR_RESOLVED!:/workspace" ^
      -v "%USERPROFILE%\.cache\hf:/home/appuser/.cache/hf" ^
      -v "%USERPROFILE%\.local\share\stt-faster:/home/appuser/.local/share/stt-faster" ^
      stt-faster:latest process /workspace --preset et-32 --language et --output-format txt --diarize --num-speakers 2 --variants "!VARIANTS_COMMA!"
)

echo.
echo ========================================
echo Processing Complete!
echo ========================================
echo Processed: !STT_AUDIO_DIR_RESOLVED!\processed\
echo Failed:    !STT_AUDIO_DIR_RESOLVED!\failed\
pause
