@echo off
REM Print the transcription queue status (no audio dir needed).

setlocal enabledelayedexpansion
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
    wsl -e bash -c "cd !STT_WSL_REPO! && .venv/bin/python scripts/transcribe_manager.py status"
) else (
    pushd "!STT_REPO_WIN!"
    .venv\Scripts\python scripts\transcribe_manager.py status
    popd
)

echo.
pause
