@echo off
REM Show recent transcription run history via `stt-faster db recent` (no audio dir needed).

setlocal enabledelayedexpansion
REM Clear any inherited STT_CALLER_DIR so _runtime.bat suppresses the
REM `audio:` banner line — this bat is a runtime-only status query.
set "STT_CALLER_DIR="
call "%~dp0_runtime.bat"
if errorlevel 1 (
    pause
    exit /b 1
)

echo ========================================
echo Transcription Status Check
echo ========================================
echo.

if /i "!STT_RUNTIME!"=="wsl" (
    wsl -e bash -c "cd !STT_WSL_REPO! && .venv/bin/stt-faster db recent"
) else (
    docker run --rm --entrypoint stt-faster ^
      -v "%USERPROFILE%\.local\share\stt-faster:/home/appuser/.local/share/stt-faster" ^
      stt-faster:latest db recent
)

echo.
pause
