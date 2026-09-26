@echo off
REM _docker_run.bat - compose docker env + run the production image.
REM
REM Caller contract:
REM   - `setlocal enabledelayedexpansion` before calling.
REM   - Caller (typically _transcribe.bat) has run _runtime.bat + _variants.bat
REM     already, so these are populated:
REM       STT_AUDIO_DIR_RESOLVED  (Windows path; mounted at /workspace)
REM       STT_DIARIZE_ARGS        (e.g. "--diarize --num-speakers 2" or "--no-diarize"; always explicit)
REM       VARIANTS_COMMA          ("52" / "1,36,44" / ...)
REM   - Caller sets STT_CLI_TAIL with the preset/language/output-format flags
REM     (e.g. "--preset turbo --language en --output-format txt").
REM   - STT_DIARIZE_ON / STT_WSL_REPO (from _runtime.bat) drive the model mounts.
REM     Host prep, in WSL: make diarization-model rnnoise-model
REM   - Helper does NOT setlocal.
REM
REM Scratch vars (leak to caller scope - reserved name, do not reuse):
REM   DOCKER_ENV_ARGS - composed `-e KEY=VALUE` flags for the docker run line.
REM     Note: this name is intentionally NOT STT_-prefixed for grep parity with
REM     the lifted source; future callers should avoid the name regardless.
REM   DOCKER_MODEL_ARGS, STT_DIAR_REPO_WIN, STT_RNNOISE_WIN - model mount scratch.
REM
REM Exits with the underlying `docker run` errorlevel (or 1 on the model gate).
REM
REM Env-bridging rationale: docker run starts with an empty env, so only the
REM Windows-side vars we explicitly pass through reach the container. WSL
REM inherits the WSL user's env (handled in _transcribe.bat's wsl branch).

REM ---- HF env pass-through (only forward what is already defined) ----
set "DOCKER_ENV_ARGS="
if defined HF_TOKEN set "DOCKER_ENV_ARGS=!DOCKER_ENV_ARGS! -e HF_TOKEN=!HF_TOKEN!"
if defined HF_XET_HIGH_PERFORMANCE set "DOCKER_ENV_ARGS=!DOCKER_ENV_ARGS! -e HF_XET_HIGH_PERFORMANCE=!HF_XET_HIGH_PERFORMANCE!"

REM Whisper model downloads only; diarization needs no token.
if defined HF_TOKEN (echo [stt-faster] HF token: set ^(passed to container^)) else (echo [stt-faster] HF token: not set ^(rate-limit warnings expected on first model download^))

REM ---- Offline models: mounted read-only from WSL, never downloaded in the container ----
REM The whole HF repo dir is mounted: snapshot files are symlinks into ../../blobs.
set "DOCKER_MODEL_ARGS="
set "STT_DIAR_REPO_WIN="
set "STT_RNNOISE_WIN="
for /f "usebackq delims=" %%p in (`wsl -e wslpath -w "!STT_WSL_REPO!/models/sh.rnnn" 2^>nul`) do set "STT_RNNOISE_WIN=%%p"

if "!STT_DIARIZE_ON!"=="1" (
    REM Checked inside WSL: config.yaml is a Linux symlink into ../../blobs.
    REM pragma: allowlist nextline secret
    wsl -e bash -c "test -f \"$HOME/.cache/hf/hub/models--pyannote-community--speaker-diarization-community-1/snapshots/8a527374977391da736e0daaef26855d949d9685/config.yaml\""
    if errorlevel 1 (
        echo [stt-faster] ERROR: speaker model not installed in WSL.
        echo [stt-faster] Run once in WSL:  cd !STT_WSL_REPO! ^&^& make diarization-model
        exit /b 1
    )
    for /f "usebackq delims=" %%p in (`wsl -e bash -c "wslpath -w \"$HOME/.cache/hf/hub/models--pyannote-community--speaker-diarization-community-1\""`) do set "STT_DIAR_REPO_WIN=%%p"
    REM pragma: allowlist nextline secret
    set "DOCKER_MODEL_ARGS=-v "!STT_DIAR_REPO_WIN!:/models/diarization:ro" -e STT_DIARIZATION_MODEL_DIR=/models/diarization/snapshots/8a527374977391da736e0daaef26855d949d9685"
)

if exist "!STT_RNNOISE_WIN!" (
    set "DOCKER_MODEL_ARGS=!DOCKER_MODEL_ARGS! -v "!STT_RNNOISE_WIN!:/models/sh.rnnn:ro" -e STT_PREPROCESS_RNNOISE_MODEL=/models/sh.rnnn"
) else (
    echo [stt-faster] RNNoise model not found in WSL - run: make rnnoise-model
)

docker run --rm ^
  -v "!STT_AUDIO_DIR_RESOLVED!:/workspace" ^
  -v "%USERPROFILE%\.cache\hf:/home/appuser/.cache/hf" ^
  -v "%USERPROFILE%\.local\share\stt-faster:/home/appuser/.local/share/stt-faster" ^
  !DOCKER_ENV_ARGS! !DOCKER_MODEL_ARGS! ^
  stt-faster:latest process /workspace !STT_CLI_TAIL! !STT_DIARIZE_ARGS! --variants "!VARIANTS_COMMA!"

exit /b !errorlevel!
