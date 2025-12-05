@echo off
REM Audio Transcription - ENGLISH/Multilingual
REM Uses: Whisper large-v3 model (large8gb preset)

echo ========================================
echo Audio Transcription - ENGLISH
echo ========================================
echo.
echo Model: Whisper large-v3 (8GB optimized)
echo Language: English / Multilingual
echo Processing audio files in: %~dp0
echo.

REM Run the transcription script via WSL with large8gb preset
wsl -e bash -c "cd ~/projects/stt-faster && .venv/bin/python scripts/transcribe_manager.py process /mnt/c/Users/PC/Downloads/transcribe --preset large8gb"

echo.
echo ========================================
echo Processing Complete!
echo ========================================
echo.
echo Processed files are in: %~dp0processed\
echo Failed files are in: %~dp0failed\
echo.

pause

