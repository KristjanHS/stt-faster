@echo off
REM Audio Transcription - Run every builtin variant against a single audio file
REM and regenerate the HTML variant comparison report.

setlocal enabledelayedexpansion
set "STT_CALLER_DIR=%~dp0"
call "%~dp0_runtime.bat"
if errorlevel 1 (
    pause
    exit /b 1
)

REM Find an audio file in the resolved audio directory.
set "AUDIO_FILE="
for %%f in ("!STT_AUDIO_DIR_RESOLVED!\*.aac" "!STT_AUDIO_DIR_RESOLVED!\*.wav" "!STT_AUDIO_DIR_RESOLVED!\*.mp3" "!STT_AUDIO_DIR_RESOLVED!\*.m4a" "!STT_AUDIO_DIR_RESOLVED!\*.flac" "!STT_AUDIO_DIR_RESOLVED!\*.ogg") do (
    if not defined AUDIO_FILE set "AUDIO_FILE=%%f"
)
if not defined AUDIO_FILE (
    echo No audio file found in !STT_AUDIO_DIR_RESOLVED!
    echo Supported formats: .wav, .mp3, .m4a, .flac, .ogg, .aac
    pause
    exit /b 1
)
echo Audio file: !AUDIO_FILE!
echo.

REM Get builtin variants from Python via a temp file (captures stdout).
set "TEMP_VARIANTS_FILE=%TEMP%\variants_output_%RANDOM%.txt"
set "VARIANT_PY=from backend.variants.registry import get_builtin_variants; variants = get_builtin_variants(); print(','.join(str(v.number) for v in variants))"

echo Getting builtin variants...
if /i "!STT_RUNTIME!"=="wsl" (
    wsl -e bash -c "cd !STT_WSL_REPO! && .venv/bin/python -c \"!VARIANT_PY!\"" > "!TEMP_VARIANTS_FILE!" 2>&1
) else (
    docker run --rm --entrypoint python stt-faster:latest -c "!VARIANT_PY!" > "!TEMP_VARIANTS_FILE!" 2>&1
)
set "PYTHON_ERROR=!errorlevel!"

if !PYTHON_ERROR! neq 0 (
    echo.
    echo ========================================
    echo ERROR: Failed to get builtin variants (error code !PYTHON_ERROR!)
    echo ========================================
    echo.
    type "!TEMP_VARIANTS_FILE!"
    del "!TEMP_VARIANTS_FILE!" >nul 2>&1
    echo.
    pause
    exit /b !PYTHON_ERROR!
)

set "VARIANTS="
REM Quote the path: %TEMP% is typically C:\Users\<user>\AppData\Local\Temp;
REM unquoted `for /f` silently fails when the username contains a space.
for /f "delims=" %%i in ("!TEMP_VARIANTS_FILE!") do set "VARIANTS=%%i"
del "!TEMP_VARIANTS_FILE!" >nul 2>&1

if "!VARIANTS!"=="" (
    echo.
    echo ========================================
    echo ERROR: No variants returned by Python.
    echo ========================================
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Builtin Variant Comparison
echo ========================================
echo Model:    TalTech Estonian Whisper (default: et-subs)
echo Language: Estonian
echo Variants: !VARIANTS!
echo.
echo Starting transcription...
echo NOTE: You will see real-time progress for each variant as it processes.
echo.

if /i "!STT_RUNTIME!"=="wsl" (
    wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && cd !STT_WSL_REPO! && .venv/bin/python scripts/transcribe_manager.py process '!STT_AUDIO_DIR_WSL!' --language et --output-format both --no-diarize --variants '!VARIANTS!'"
) else (
    docker run --rm ^
      -v "!STT_AUDIO_DIR_RESOLVED!:/workspace" ^
      -v "%USERPROFILE%\.cache\hf:/home/appuser/.cache/hf" ^
      -v "%USERPROFILE%\.local\share\stt-faster:/home/appuser/.local/share/stt-faster" ^
      stt-faster:latest process /workspace --language et --output-format both --no-diarize --variants "!VARIANTS!"
)
set "TRANSCRIBE_ERROR=!errorlevel!"

if !TRANSCRIBE_ERROR! neq 0 (
    echo.
    echo ========================================
    echo ERROR: Transcription failed with error code !TRANSCRIBE_ERROR!
    echo ========================================
    echo.
    pause
    exit /b !TRANSCRIBE_ERROR!
)

echo.
echo ========================================
echo Processing Complete!
echo ========================================
echo Outputs are organized in: !STT_AUDIO_DIR_RESOLVED!\variant_outputs\
echo Each variant has its own folder with processed/ and run_meta.json
echo.
echo ========================================
echo Generating variant report...
echo ========================================
echo.

if /i "!STT_RUNTIME!"=="wsl" (
    wsl -e bash -c "cd !STT_WSL_REPO! && .venv/bin/python scripts/generate_variant_report.py --far-speaker-range 252-291 --silence-range 19-61"
) else (
    docker run --rm ^
      -v "!STT_AUDIO_DIR_RESOLVED!:/workspace" ^
      -v "%USERPROFILE%\.local\share\stt-faster:/home/appuser/.local/share/stt-faster" ^
      --entrypoint python ^
      stt-faster:latest /app/scripts/generate_variant_report.py --far-speaker-range 252-291 --silence-range 19-61
)
set "REPORT_ERROR=!errorlevel!"

if !REPORT_ERROR! neq 0 (
    echo.
    echo WARNING: Variant report generation failed with error code !REPORT_ERROR!
    echo.
) else (
    echo.
    echo Variant report generated successfully.
    echo.
)

pause
