@echo off
REM _transcribe.bat - dispatch wsl vs docker after _runtime.bat + _variants.bat.
REM
REM Caller contract:
REM   - `setlocal enabledelayedexpansion` before calling.
REM   - Caller already invoked _runtime.bat (populates STT_RUNTIME,
REM     STT_DIARIZE_ARGS, STT_AUDIO_DIR_RESOLVED, STT_AUDIO_DIR_WSL,
REM     STT_WSL_REPO) and _variants.bat (populates VARIANTS_COMMA).
REM   - Caller sets STT_CLI_TAIL with preset/language/output-format flags
REM     (e.g. "--preset turbo --language en --output-format txt"). Shared
REM     between the WSL and docker branches today; collapses both branches
REM     to one knob.
REM   - DIARIZE (raw caller knob) is read transitively by _docker_run.bat
REM     for the HF_TOKEN fail-fast guard - caller must set it (or leave
REM     unset for the runtime-default off-for-docker behavior) before
REM     calling this helper.
REM   - Helper does NOT setlocal.
REM
REM Env-bridging asymmetry (persists by design):
REM   - WSL branch inherits the WSL user's env and exports HF_HOME inline.
REM   - Docker branch starts with an empty env; _docker_run.bat forwards
REM     HF_TOKEN / HF_XET_HIGH_PERFORMANCE explicitly via -e flags.

if /i "!STT_RUNTIME!"=="wsl" (
    wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && cd !STT_WSL_REPO! && .venv/bin/python scripts/transcribe_manager.py process '!STT_AUDIO_DIR_WSL!' !STT_CLI_TAIL! !STT_DIARIZE_ARGS! --variants '!VARIANTS_COMMA!'"
    exit /b !errorlevel!
)

call "%~dp0_docker_run.bat"
exit /b !errorlevel!
