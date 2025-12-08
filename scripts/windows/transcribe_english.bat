@echo off
REM Audio Transcription - ENGLISH/Multilingual
REM Uses: Whisper large-v3 model (large8gb preset)
REM Note: Uses HF cache at ~/.cache/hf/ for faster model loading

echo ========================================
echo Audio Transcription - ENGLISH
echo ========================================
echo.
echo Model: Whisper Turbo (distil-large-v3)
echo Language: English / Multilingual
echo Pre-processing: enabled (downmix/resample + loudnorm)
echo Processing audio files in: %~dp0
echo.

REM Run the transcription script via WSL with turbo preset
REM Set HF_HOME to use the correct cache location
REM Force English language to avoid auto-detection errors
REM Enable light audio pre-processing (downmix/resample + loudnorm)
REM Output both txt and json formats
wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && export STT_PREPROCESS_ENABLED=1 && cd ~/projects/stt-faster && .venv/bin/python scripts/transcribe_manager.py process /mnt/c/Users/PC/Downloads/transcribe --preset turbo --language en --output-format both"

echo.
echo ========================================
echo Processing Complete!
echo ========================================
echo.
echo Processed files are in: %~dp0processed\
echo Failed files are in: %~dp0failed\
echo.

pause
