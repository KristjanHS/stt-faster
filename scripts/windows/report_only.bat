@echo off
REM Regenerate the variant HTML report from existing outputs (no transcription).

setlocal enabledelayedexpansion
REM Clear any inherited STT_CALLER_DIR so _runtime.bat suppresses the
REM `audio:` banner line — this bat regenerates a report from existing
REM outputs and doesn't consume the audio dir.
set "STT_CALLER_DIR="
call "%~dp0_runtime.bat"
if errorlevel 1 (
    pause
    exit /b 1
)

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
    echo ========================================
    echo WARNING: Variant report generation failed with error code !REPORT_ERROR!
    echo ========================================
    echo.
    echo Check the output above for error details.
    echo.
) else (
    echo.
    echo Variant report generated successfully.
    echo.
)

pause
