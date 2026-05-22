@echo off
REM Audio Transcription - ENGLISH, Online / mixed-source (variant 52, turbo preset)
REM ** Docker-forced variant **: bypasses _runtime.bat's WSL probe and always
REM runs against the stt-faster:latest production image. Use to exercise the
REM docker code path on a machine where the WSL venv is also available.
REM
REM Prereq: image must exist on the Docker daemon. Build via WSL:
REM     wsl -- bash -lc "cd /home/kristjans/projects/stt-faster && make docker-build-prod"
REM Or directly:  docker build -t stt-faster:latest <repo>
REM
REM Optional Windows env vars (set once with `setx NAME value`, then reopen cmd):
REM   HF_TOKEN                  - HF Hub auth. Silences rate-limit warnings;
REM                               REQUIRED if DIARIZE=1 (pyannote is gated).
REM   HF_XET_HIGH_PERFORMANCE   - "1" turns on Xet high-performance downloads
REM                               (the `hf_xet` package is already in the image).
REM                               Replaces the deprecated HF_HUB_ENABLE_HF_TRANSFER
REM                               flag from huggingface_hub <1.x.
REM Both are passed through to the container only if defined here.
set "VARIANTS=52"
set "DIARIZE="          REM "" = off (docker default); "0" = off; "1" = on
set "NUM_SPEAKERS=2"    REM only used when DIARIZE=1; "" = auto-detect

setlocal enabledelayedexpansion

REM ---- Audio dir resolution (mirror of _runtime.bat's STT_AUDIO_DIR_RESOLVED) ----
REM Precedence: STT_AUDIO_DIR env var > bat's own folder > %CD%.
set "STT_AUDIO_DIR_RESOLVED="
if defined STT_AUDIO_DIR set "STT_AUDIO_DIR_RESOLVED=%STT_AUDIO_DIR%"
if not defined STT_AUDIO_DIR_RESOLVED set "STT_AUDIO_DIR_RESOLVED=%~dp0"
if "%STT_AUDIO_DIR_RESOLVED:~-1%"=="\" set "STT_AUDIO_DIR_RESOLVED=%STT_AUDIO_DIR_RESOLVED:~0,-1%"

REM ---- Image pre-flight: offer to build if missing ----
REM Auto-build only works when this .bat sits in its original repo location
REM (i.e. <repo>/scripts/windows/). When copied next to audio, the repo-detect
REM fails and we fall back to printing the manual command.
REM
REM Structured as flat goto-labels rather than nested if/else: CMD's parser
REM counts parens at parse-time, and caret-escaped parens (`^(`, `^)`) inside
REM echoes within a 3-level parenthesized block become unreliable — both the
REM IF and ELSE branches end up running. Flat form avoids the trap entirely.
docker image inspect stt-faster:latest >nul 2>nul
if not errorlevel 1 goto :image_ready

echo [stt-faster] Image stt-faster:latest not found on Docker daemon.
set "STT_REPO_GUESS=%~dp0..\.."
if not exist "!STT_REPO_GUESS!\Dockerfile" goto :no_repo

echo [stt-faster] Repo detected at: !STT_REPO_GUESS!
choice /M "[stt-faster] Build it now (~5-10 min, downloads layers)" /C YN
REM Explicit Y-only: errorlevel 1 = Y, 2 = N, 0 = Ctrl+C, 255 = choice error.
REM Anything other than 1 is treated as abort, including non-console stdin.
set "STT_CHOICE_RC=!errorlevel!"
if not "!STT_CHOICE_RC!"=="1" goto :build_declined

REM pushd to a directory we just confirmed exists via `if exist Dockerfile`,
REM so a guard on pushd would be dead code.
pushd "!STT_REPO_GUESS!"
docker build -t stt-faster:latest .
set "STT_BUILD_RC=!errorlevel!"
popd
if not "!STT_BUILD_RC!"=="0" goto :build_failed
echo [stt-faster] Build complete.
goto :image_ready

:build_declined
echo [stt-faster] Aborted. To build manually:
echo [stt-faster]   cd /d "!STT_REPO_GUESS!"
echo [stt-faster]   docker build -t stt-faster:latest .
pause
exit /b 1

:build_failed
echo [stt-faster] Build failed, rc=!STT_BUILD_RC!. Is Docker Desktop running?
pause
exit /b !STT_BUILD_RC!

:no_repo
echo [stt-faster] This .bat is not next to a Dockerfile -- probably copied out
echo [stt-faster] of the repo. Build the image manually from a cmd window
echo [stt-faster] opened in the stt-faster repo:
echo [stt-faster]   docker build -t stt-faster:latest .
pause
exit /b 1

:image_ready

REM ---- Diarize args (DIARIZE=1 forces on; anything else = off) ----
set "STT_DIARIZE_ARGS="
if "!DIARIZE!"=="1" (
    set "STT_DIARIZE_ARGS=--diarize"
    if defined NUM_SPEAKERS if not "!NUM_SPEAKERS!"=="" set "STT_DIARIZE_ARGS=--diarize --num-speakers !NUM_SPEAKERS!"
)

REM ---- HF env pass-through: docker run starts with an empty env; bridge only ----
REM ---- what's already defined on the Windows side. WSL parity not assumed.   ----
set "DOCKER_ENV_ARGS="
if defined HF_TOKEN set "DOCKER_ENV_ARGS=!DOCKER_ENV_ARGS! -e HF_TOKEN=!HF_TOKEN!"
if defined HF_XET_HIGH_PERFORMANCE set "DOCKER_ENV_ARGS=!DOCKER_ENV_ARGS! -e HF_XET_HIGH_PERFORMANCE=!HF_XET_HIGH_PERFORMANCE!"

REM Diarize is gated by HF_TOKEN — fail fast with a clear hint, don't let the
REM container start and bomb out mid-pipeline.
if "!DIARIZE!"=="1" if not defined HF_TOKEN (
    echo [stt-faster] ERROR: DIARIZE=1 but HF_TOKEN is not set on Windows.
    echo [stt-faster] Pyannote diarization requires an HF token with the
    echo [stt-faster] pyannote/speaker-diarization-3.1 license accepted.
    echo [stt-faster] Set it once:  setx HF_TOKEN ^<your-hf-token^>
    echo [stt-faster] Then reopen this cmd window and re-run.
    pause
    exit /b 1
)

REM ---- Variants: space -> comma (matches sibling bats) ----
set "VARIANTS_COMMA="
set /a VARIANT_COUNT=0
for %%v in (!VARIANTS!) do (
    if "!VARIANTS_COMMA!"=="" (
        set "VARIANTS_COMMA=%%v"
    ) else (
        set "VARIANTS_COMMA=!VARIANTS_COMMA!,%%v"
    )
    set /a VARIANT_COUNT+=1
)

echo [stt-faster] runtime: docker (forced)
echo ========================================
echo Audio Transcription - ENGLISH (Online) [DOCKER FORCED]
echo ========================================
echo Model:    Systran faster-distil-whisper-large-v3 (turbo)
echo Language: English
if !VARIANT_COUNT!==1 (echo Variant:  !VARIANTS_COMMA!) else (echo Variants: !VARIANTS_COMMA! ^(!VARIANT_COUNT! variants^))
echo Audio:    !STT_AUDIO_DIR_RESOLVED!
echo HF cache: %USERPROFILE%\.cache\hf
if "!STT_DIARIZE_ARGS!"=="" (echo Diarize:  off) else (echo Diarize:  !STT_DIARIZE_ARGS!)
if defined HF_TOKEN (echo HF token: set ^(passed to container^)) else (echo HF token: not set ^(rate-limit warnings expected^))
echo.

docker run --rm ^
  -v "!STT_AUDIO_DIR_RESOLVED!:/workspace" ^
  -v "%USERPROFILE%\.cache\hf:/home/appuser/.cache/hf" ^
  -v "%USERPROFILE%\.local\share\stt-faster:/home/appuser/.local/share/stt-faster" ^
  !DOCKER_ENV_ARGS! ^
  stt-faster:latest process /workspace --preset turbo --language en --output-format txt !STT_DIARIZE_ARGS! --variants "!VARIANTS_COMMA!"

echo.
echo ========================================
echo Processing Complete!
echo ========================================
echo Processed: !STT_AUDIO_DIR_RESOLVED!\processed\
echo Failed:    !STT_AUDIO_DIR_RESOLVED!\failed\
pause
