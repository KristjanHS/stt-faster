@echo off
REM Audio Transcription - ENGLISH, Online / mixed-source (variant 52, turbo preset)
REM Edit the VARIANTS line below to use a different variant.
REM Space-separated for multiple variants (e.g. "1 36 44").
set "VARIANTS=52"
set "DIARIZE="          REM "" = runtime default (wsl=on, docker=off); "0" = off; "1" = on
set "NUM_SPEAKERS=2"    REM only used when diarize resolves to on; "" = auto-detect

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
echo Audio Transcription - ENGLISH (Online)
echo ========================================
echo Model:    Systran faster-distil-whisper-large-v3 (turbo)
echo Language: English
if !VARIANT_COUNT!==1 (echo Variant:  !VARIANTS_COMMA!) else (echo Variants: !VARIANTS_COMMA! ^(!VARIANT_COUNT! variants^))
echo.

if /i "!STT_RUNTIME!"=="wsl" (
    wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && cd !STT_WSL_REPO! && .venv/bin/python scripts/transcribe_manager.py process '!STT_AUDIO_DIR_WSL!' --preset turbo --language en --output-format txt !STT_DIARIZE_ARGS! --variants '!VARIANTS_COMMA!'"
) else (
    docker run --rm ^
      -v "!STT_AUDIO_DIR_RESOLVED!:/workspace" ^
      -v "%USERPROFILE%\.cache\hf:/home/appuser/.cache/hf" ^
      -v "%USERPROFILE%\.local\share\stt-faster:/home/appuser/.local/share/stt-faster" ^
      stt-faster:latest process /workspace --preset turbo --language en --output-format txt !STT_DIARIZE_ARGS! --variants "!VARIANTS_COMMA!"
)

echo.
echo ========================================
echo Processing Complete!
echo ========================================
echo Processed: !STT_AUDIO_DIR_RESOLVED!\processed\
echo Failed:    !STT_AUDIO_DIR_RESOLVED!\failed\
pause
