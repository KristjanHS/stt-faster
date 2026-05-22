@echo off
REM _runtime.bat - shared runtime selection + path translation for stt-faster bats.
REM
REM Caller contract:
REM   - `setlocal enabledelayedexpansion` before calling.
REM   - Optionally `set "STT_CALLER_DIR=%~dp0"` to use the bat's folder as the
REM     audio-dir fallback (when STT_AUDIO_DIR env var is not set).
REM
REM Caller may also set (optional):
REM   DIARIZE           - "" / unset = runtime default; "0" = force off; "1" = force on
REM   NUM_SPEAKERS      - speaker count for diarization; "" = auto-detect
REM   STT_FORCE_DOCKER  - "1" skips the WSL probe (docker-forced bats)
REM   STT_NO_AUTOBUILD  - "1" disables the docker-image auto-build prompt
REM
REM Outputs (visible to caller; this helper deliberately does NOT setlocal):
REM   STT_RUNTIME            - "wsl" or "docker"  (this helper exits /b 1 on "none")
REM   STT_WSL_REPO           - hardcoded WSL repo path
REM   STT_AUDIO_DIR_RESOLVED - Windows path to audio dir (trailing \ stripped)
REM   STT_AUDIO_DIR_WSL      - same path translated to /mnt/<drive>/... form
REM   STT_DIARIZE_ARGS       - composed CLI args ("--diarize --num-speakers 2", or empty)
REM   STT_REPO_GUESS         - guessed repo root (`%~dp0..\..`); used by auto-build
REM
REM CMD-parser notes:
REM   - Helpers assume the caller did `setlocal enabledelayedexpansion`;
REM     helpers do NOT `setlocal` themselves (would hide outputs from caller).
REM   - Auto-build flow uses flat `goto :label`. Never nested if/else with
REM     caret-escaped echoes (parser counts parens at parse-time and the
REM     caret-escaped `^(` `^)` inside a 3-level block become unreliable;
REM     both IF and ELSE branches can end up running).
REM   - Variable-existence checks use `if defined VAR if not "!VAR!"==""` to
REM     handle both unset and defined-but-empty cases.

REM Hardcoded WSL repo path. Matches the current dev machine; any other
REM machine fails the probe below and falls back to Docker. Edit this line
REM if the repo moves inside WSL.
set "STT_WSL_REPO=/home/kristjans/projects/stt-faster"

REM Repo root for the auto-build flow. `%~dp0..\..` resolves to <repo>/
REM (this bat sits at <repo>/scripts/windows/). When the bat is copied
REM next to audio, the `if exist Dockerfile` check below fails and the
REM flow falls back to the manual-build hint.
set "STT_REPO_GUESS=%~dp0..\.."

REM ---- Audio dir resolution ----
REM Parse-time %% is intentional below: STT_AUDIO_DIR and STT_CALLER_DIR are
REM set in the caller scope BEFORE the `call` to this helper, so their values
REM are stable. !VAR! (delayed) would also work but is not required here.
set "STT_AUDIO_DIR_RESOLVED="
if defined STT_AUDIO_DIR set "STT_AUDIO_DIR_RESOLVED=%STT_AUDIO_DIR%"
if not defined STT_AUDIO_DIR_RESOLVED if defined STT_CALLER_DIR set "STT_AUDIO_DIR_RESOLVED=%STT_CALLER_DIR%"
if not defined STT_AUDIO_DIR_RESOLVED set "STT_AUDIO_DIR_RESOLVED=%CD%"
if "%STT_AUDIO_DIR_RESOLVED:~-1%"=="\" set "STT_AUDIO_DIR_RESOLVED=%STT_AUDIO_DIR_RESOLVED:~0,-1%"

REM ---- WSL path translation (C:\foo\bar -> /mnt/c/foo/bar) ----
REM Prefix-only substring splice (not the global `%VAR:C:=/mnt/c%` form) so a
REM path that legitimately contains "C:" later (rare but possible) isn't
REM double-translated. Drive letters beyond C/D/E fall through with backslashes
REM converted but no /mnt/ prefix - matches the legacy bats' scope.
set "STT_AUDIO_DIR_WSL=%STT_AUDIO_DIR_RESOLVED:\=/%"
if /i "%STT_AUDIO_DIR_RESOLVED:~0,2%"=="C:" set "STT_AUDIO_DIR_WSL=/mnt/c%STT_AUDIO_DIR_WSL:~2%"
if /i "%STT_AUDIO_DIR_RESOLVED:~0,2%"=="D:" set "STT_AUDIO_DIR_WSL=/mnt/d%STT_AUDIO_DIR_WSL:~2%"
if /i "%STT_AUDIO_DIR_RESOLVED:~0,2%"=="E:" set "STT_AUDIO_DIR_WSL=/mnt/e%STT_AUDIO_DIR_WSL:~2%"

REM ---- Runtime probes: WSL first (unless forced docker), then Docker ----
REM WSL probe: WSL installed AND the hardcoded repo has a usable .venv.
REM Double-quote the WSL-side path: harmless on space-free paths (today's
REM hardcoded value) and defensible if the constant is ever edited to a path
REM containing spaces.
set "STT_RUNTIME=none"
if /i not "%STT_FORCE_DOCKER%"=="1" (
    where wsl >nul 2>nul && wsl -e test -x "%STT_WSL_REPO%/.venv/bin/python" >nul 2>nul && set "STT_RUNTIME=wsl"
)

REM Docker probe: docker on PATH AND the production image is already built.
REM `docker image inspect` (not `docker info`) - daemon-running AND
REM image-built must both hold. A daemon-only probe would pass and then
REM `docker run` would fail with a worse error.
if /i "%STT_RUNTIME%"=="none" (
    where docker >nul 2>nul && docker image inspect stt-faster:latest >nul 2>nul && set "STT_RUNTIME=docker"
)

REM ---- Auto-build: docker present, image missing, autobuild not opted-out ----
REM Verbatim port of transcribe_english_Online_docker.bat:41-83 (flat goto
REM form for the same parser-trap reason). Skips entirely if WSL already
REM resolved, if docker is not on PATH, or if STT_NO_AUTOBUILD=1.
if /i not "%STT_RUNTIME%"=="none" goto :runtime_after_autobuild
if /i "%STT_NO_AUTOBUILD%"=="1" goto :runtime_after_autobuild
where docker >nul 2>nul
if errorlevel 1 goto :runtime_after_autobuild

echo [stt-faster] Image stt-faster:latest not found on Docker daemon.
if not exist "!STT_REPO_GUESS!\Dockerfile" goto :runtime_no_repo

echo [stt-faster] Repo detected at: !STT_REPO_GUESS!
choice /M "[stt-faster] Build it now (~5-10 min, downloads layers)" /C YN
REM Explicit Y-only: errorlevel 1 = Y, 2 = N, 0 = Ctrl+C, 255 = choice error.
REM Anything other than 1 is treated as abort, including non-console stdin.
set "STT_CHOICE_RC=!errorlevel!"
if not "!STT_CHOICE_RC!"=="1" goto :runtime_build_declined

REM pushd to a directory we just confirmed exists via `if exist Dockerfile`,
REM so a guard on pushd would be dead code.
pushd "!STT_REPO_GUESS!"
docker build -t stt-faster:latest .
set "STT_BUILD_RC=!errorlevel!"
popd
if not "!STT_BUILD_RC!"=="0" goto :runtime_build_failed
echo [stt-faster] Build complete.
set "STT_RUNTIME=docker"
goto :runtime_after_autobuild

:runtime_build_declined
echo [stt-faster] Aborted. To build manually:
echo [stt-faster]   cd /d "!STT_REPO_GUESS!"
echo [stt-faster]   docker build -t stt-faster:latest .
exit /b 1

:runtime_build_failed
echo [stt-faster] Build failed, rc=!STT_BUILD_RC!. Is Docker Desktop running?
exit /b !STT_BUILD_RC!

:runtime_no_repo
echo [stt-faster] This .bat is not next to a Dockerfile -- probably copied out
echo [stt-faster] of the repo. Build the image manually from a cmd window
echo [stt-faster] opened in the stt-faster repo:
echo [stt-faster]   docker build -t stt-faster:latest .
exit /b 1

:runtime_after_autobuild

REM ---- Diarize resolution: caller knobs + runtime-aware default ----
REM   DIARIZE=1        -> force on (any runtime)
REM   DIARIZE=0        -> force off (any runtime)
REM   DIARIZE unset/"" -> on if STT_RUNTIME=wsl, off otherwise
REM `set "DIARIZE="` may either undefine DIARIZE or leave it defined-but-empty
REM depending on the Windows/CMD build; lines below handle both cases.
REM `!NUM_SPEAKERS!` (delayed) inside the parens block below avoids the
REM parse-time-expansion trap with caller-scope vars.
set "STT_DIARIZE_ARGS="
set "STT_DIARIZE_ON=0"
if /i "%DIARIZE%"=="1" set "STT_DIARIZE_ON=1"
if not defined DIARIZE if /i "%STT_RUNTIME%"=="wsl" set "STT_DIARIZE_ON=1"
if defined DIARIZE if "%DIARIZE%"=="" if /i "%STT_RUNTIME%"=="wsl" set "STT_DIARIZE_ON=1"
if /i "%DIARIZE%"=="0" set "STT_DIARIZE_ON=0"

if "%STT_DIARIZE_ON%"=="1" (
    set "STT_DIARIZE_ARGS=--diarize"
    if defined NUM_SPEAKERS if not "!NUM_SPEAKERS!"=="" set "STT_DIARIZE_ARGS=--diarize --num-speakers !NUM_SPEAKERS!"
)

REM ---- Banner (mirrors the GPU-fallback banner philosophy) ----
REM Skip the `audio:` line when the caller never asked us to resolve one
REM (utility bats like check_recent that don't process audio). The %CD%
REM fallback above still keeps STT_AUDIO_DIR_RESOLVED non-empty for any
REM downstream code that reads it; the banner just stops being misleading.
echo [stt-faster] runtime: %STT_RUNTIME%
if defined STT_AUDIO_DIR (
    echo [stt-faster] audio:   %STT_AUDIO_DIR_RESOLVED%
) else if defined STT_CALLER_DIR (
    echo [stt-faster] audio:   %STT_AUDIO_DIR_RESOLVED%
)
if /i "%STT_RUNTIME%"=="wsl" echo [stt-faster] WSL repo: %STT_WSL_REPO%
REM Compose banner string in flat sequential ifs - nested if/else inside a
REM parens block is parser-fragile in CMD (else-binding ambiguity).
set "STT_DIARIZE_BANNER=off"
if "%STT_DIARIZE_ON%"=="1" set "STT_DIARIZE_BANNER=on"
if "%STT_DIARIZE_ON%"=="1" if defined NUM_SPEAKERS if not "!NUM_SPEAKERS!"=="" set "STT_DIARIZE_BANNER=on (--num-speakers !NUM_SPEAKERS!)"
echo [stt-faster] diarize: %STT_DIARIZE_BANNER%

REM ---- Failure case: WSL absent + docker absent / image missing + autobuild ----
REM disabled/declined/failed. Auto-build flow exits /b on its own failure
REM paths; this generic message only fires when nothing on the box could run.
if /i "%STT_RUNTIME%"=="none" (
    echo [stt-faster] ERROR: no runtime available.
    echo [stt-faster] Install Docker Desktop and run setup.bat to build the image.
    exit /b 1
)

exit /b 0
