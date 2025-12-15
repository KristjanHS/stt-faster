@echo off
REM Check transcription status

echo ========================================
echo Transcription Status Check
echo ========================================
echo.

REM Run the status command via WSL
wsl -e bash -c "cd ~/projects/stt-faster && .venv/bin/python scripts/transcribe_manager.py status"

echo.
pause










