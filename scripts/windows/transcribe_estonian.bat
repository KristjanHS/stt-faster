@echo off
REM Audio Transcription - ESTONIAN Language
REM Uses: TalTech Estonian Whisper model (et-large preset)
REM Note: Uses HF cache at ~/.cache/hf/ for faster model loading

echo ========================================
echo Audio Transcription - ESTONIAN
echo ========================================
echo.
echo Model: TalTech Estonian Whisper (large-v3-turbo)
echo Language: Estonian
echo Pre-processing: enabled (downmix/resample + loudnorm)
echo Processing audio files in: %~dp0
echo.

REM Run the transcription script via WSL (uses default et-large preset)
REM Set HF_HOME to use the correct cache location
REM Force Estonian language to avoid auto-detection errors
REM Enable light audio pre-processing (downmix/resample + loudnorm)
wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && export STT_PREPROCESS_ENABLED=1 && cd ~/projects/stt-faster && .venv/bin/python scripts/transcribe_manager.py process /mnt/c/Users/PC/Downloads/transcribe --language et"

echo.
echo ========================================
echo Processing Complete!
echo ========================================
echo.
echo Processed files are in: %~dp0processed\
echo Failed files are in: %~dp0failed\
echo.

pause
