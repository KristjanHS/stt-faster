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
REM   - DIARIZE (raw caller knob) is read for the HF_TOKEN fail-fast guard.
REM   - Helper does NOT setlocal.
REM
REM Scratch vars (leak to caller scope - reserved name, do not reuse):
REM   DOCKER_ENV_ARGS - composed `-e KEY=VALUE` flags for the docker run line.
REM     Note: this name is intentionally NOT STT_-prefixed for grep parity with
REM     the lifted source; future callers should avoid the name regardless.
REM
REM Exits with the underlying `docker run` errorlevel (or 1 on the HF gate).
REM
REM Env-bridging rationale: docker run starts with an empty env, so only the
REM Windows-side vars we explicitly pass through reach the container. WSL
REM inherits the WSL user's env (handled in _transcribe.bat's wsl branch).

REM ---- HF env pass-through (only forward what is already defined) ----
set "DOCKER_ENV_ARGS="
if defined HF_TOKEN set "DOCKER_ENV_ARGS=!DOCKER_ENV_ARGS! -e HF_TOKEN=!HF_TOKEN!"
if defined HF_XET_HIGH_PERFORMANCE set "DOCKER_ENV_ARGS=!DOCKER_ENV_ARGS! -e HF_XET_HIGH_PERFORMANCE=!HF_XET_HIGH_PERFORMANCE!"

REM Banner-line for HF_TOKEN state - lifted from the old docker-forced bat.
REM Useful for debugging "why are model downloads slow / rate-limited" without
REM having to inspect the env. The diarize fail-fast above covers the hard
REM failure (DIARIZE=1 + missing token); this echo covers the soft case
REM (rate-limit warnings when downloading models without an authenticated session).
if defined HF_TOKEN (echo [stt-faster] HF token: set ^(passed to container^)) else (echo [stt-faster] HF token: not set ^(rate-limit warnings expected on first model download^))

REM ---- Diarize fail-fast: pyannote requires HF_TOKEN with the license accepted ----
REM Fire before docker run so the user gets a clear hint instead of a
REM mid-pipeline crash inside the container. Trigger on the raw caller
REM knob DIARIZE=1 (matches the lifted source); _runtime.bat's runtime-
REM default off-for-docker case leaves DIARIZE unset and skips this branch.
if "!DIARIZE!"=="1" if not defined HF_TOKEN (
    echo [stt-faster] ERROR: DIARIZE=1 but HF_TOKEN is not set on Windows.
    echo [stt-faster] Pyannote diarization requires an HF token with the
    echo [stt-faster] pyannote/speaker-diarization-3.1 license accepted.
    echo [stt-faster] Set it once:  setx HF_TOKEN ^<your-hf-token^>
    echo [stt-faster] Then reopen this cmd window and re-run.
    exit /b 1
)

docker run --rm ^
  -v "!STT_AUDIO_DIR_RESOLVED!:/workspace" ^
  -v "%USERPROFILE%\.cache\hf:/home/appuser/.cache/hf" ^
  -v "%USERPROFILE%\.local\share\stt-faster:/home/appuser/.local/share/stt-faster" ^
  !DOCKER_ENV_ARGS! ^
  stt-faster:latest process /workspace !STT_CLI_TAIL! !STT_DIARIZE_ARGS! --variants "!VARIANTS_COMMA!"

exit /b !errorlevel!
