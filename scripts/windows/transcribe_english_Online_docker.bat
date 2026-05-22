@echo off
REM Audio Transcription - ENGLISH, Online / mixed-source (variant 52, turbo preset)
REM ** Docker-forced variant **: STT_FORCE_DOCKER=1 makes _runtime.bat skip the
REM WSL probe and always pick the stt-faster:latest production image. Use to
REM exercise the docker code path on a machine where the WSL venv is also
REM available.
REM
REM Prereq: image must exist on the Docker daemon. _runtime.bat now auto-prompts
REM to build it if missing (Stage 1 of the bat refactor). Manual build:
REM     wsl -- bash -lc "cd /home/kristjans/projects/stt-faster && make docker-build-prod"
REM Or directly:  docker build -t stt-faster:latest <repo>
REM
REM Optional Windows env vars (set once with `setx NAME value`, then reopen cmd):
REM   HF_TOKEN                  - HF Hub auth. Silences rate-limit warnings;
REM                               REQUIRED if DIARIZE=1 (pyannote is gated).
REM   HF_XET_HIGH_PERFORMANCE   - "1" turns on Xet high-performance downloads
REM                               (the `hf_xet` package is already in the image).
REM Both are passed through to the container by _docker_run.bat only if defined.
set "VARIANTS=52"
set "DIARIZE="          REM "" = off (docker default); "0" = off; "1" = on
set "NUM_SPEAKERS=2"    REM only used when DIARIZE=1; "" = auto-detect

setlocal enabledelayedexpansion
set "STT_CALLER_DIR=%~dp0"
set "STT_FORCE_DOCKER=1"
set "STT_TITLE=Audio Transcription - ENGLISH (Online)"
set "STT_TITLE_SUFFIX=[DOCKER FORCED]"
set "STT_MODEL=Systran faster-distil-whisper-large-v3 (turbo)"
set "STT_LANG=English"
set "STT_CLI_TAIL=--preset turbo --language en --output-format txt"
call "%~dp0_runtime.bat" || ( pause & exit /b 1 )
call "%~dp0_variants.bat"
call "%~dp0_banner.bat"
call "%~dp0_transcribe.bat"
call "%~dp0_footer.bat"
