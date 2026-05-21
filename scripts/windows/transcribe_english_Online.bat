@echo off
REM Audio Transcription - ENGLISH, Online / mixed-source (variant 52, turbo preset)
REM Edit the VARIANTS line below to use a different variant.
REM Space-separated for multiple variants (e.g. "1 36 44").
set "VARIANTS=52"

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
    wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && cd !STT_WSL_REPO! && .venv/bin/python scripts/transcribe_manager.py process '!STT_AUDIO_DIR_WSL!' --preset turbo --language en --output-format both --variants '!VARIANTS_COMMA!'"
) else (
    set "HF_HOME=%USERPROFILE%\.cache\hf"
    set "HF_HUB_CACHE=%USERPROFILE%\.cache\hf\hub"
    pushd "!STT_REPO_WIN!"
    .venv\Scripts\python scripts\transcribe_manager.py process "!STT_AUDIO_DIR_RESOLVED!" --preset turbo --language en --output-format both --variants "!VARIANTS_COMMA!"
    popd
)

echo.
echo ========================================
echo Processing Complete!
echo ========================================
echo Processed: !STT_AUDIO_DIR_RESOLVED!\processed\
echo Failed:    !STT_AUDIO_DIR_RESOLVED!\failed\
pause
