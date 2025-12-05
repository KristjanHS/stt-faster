@echo off
REM Audio Transcription - ESTONIAN Language
REM Uses: TalTech Estonian Whisper model (et-large preset)

echo ========================================
echo Audio Transcription - ESTONIAN
echo ========================================
echo.
echo Model: TalTech Estonian Whisper (large-v3-turbo)
echo Language: Estonian
echo Processing audio files in: %~dp0
echo.

REM Run the transcription script via WSL (uses default et-large preset)
wsl -e bash -c "cd ~/projects/stt-faster && .venv/bin/python scripts/transcribe_manager.py process /mnt/c/Users/PC/Downloads/transcribe"

echo.
echo ========================================
echo Processing Complete!
echo ========================================
echo.
echo Processed files are in: %~dp0processed\
echo Failed files are in: %~dp0failed\
echo.

pause

