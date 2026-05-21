@echo off
REM _runtime.bat - shared runtime selection + path translation for stt-faster bats.
REM
REM Caller contract:
REM   - `setlocal enabledelayedexpansion` before calling.
REM   - Optionally `set "STT_CALLER_DIR=%~dp0"` to use the bat's folder as the
REM     audio-dir fallback (when STT_AUDIO_DIR env var is not set).
REM
REM Outputs (visible to caller; this helper deliberately does NOT setlocal):
REM   STT_RUNTIME            - "wsl" or "native"
REM   STT_REPO_WIN           - absolute Windows path to repo root
REM   STT_WSL_REPO           - hardcoded WSL repo path
REM   STT_AUDIO_DIR_RESOLVED - Windows path to audio dir (trailing \ stripped)
REM   STT_AUDIO_DIR_WSL      - same path translated to /mnt/<drive>/... form

REM Hardcoded WSL repo path. Matches the current dev machine; any other
REM machine fails the probe below and falls back to native. Edit this line
REM if the repo moves inside WSL.
set "STT_WSL_REPO=/home/kristjans/projects/stt-faster"

REM Repo root: _runtime.bat lives at <repo>\scripts\windows\_runtime.bat
for %%I in ("%~dp0..\..") do set "STT_REPO_WIN=%%~fI"

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
REM converted but no /mnt/ prefix — matches the legacy bats' scope.
set "STT_AUDIO_DIR_WSL=%STT_AUDIO_DIR_RESOLVED:\=/%"
if /i "%STT_AUDIO_DIR_RESOLVED:~0,2%"=="C:" set "STT_AUDIO_DIR_WSL=/mnt/c%STT_AUDIO_DIR_WSL:~2%"
if /i "%STT_AUDIO_DIR_RESOLVED:~0,2%"=="D:" set "STT_AUDIO_DIR_WSL=/mnt/d%STT_AUDIO_DIR_WSL:~2%"
if /i "%STT_AUDIO_DIR_RESOLVED:~0,2%"=="E:" set "STT_AUDIO_DIR_WSL=/mnt/e%STT_AUDIO_DIR_WSL:~2%"

REM ---- WSL probe: WSL installed AND the hardcoded repo has a usable .venv ----
REM Double-quote the WSL-side path: harmless on space-free paths (today's
REM hardcoded value) and defensible if the constant is ever edited to a path
REM containing spaces.
set "STT_RUNTIME=native"
where wsl >nul 2>nul && wsl -e test -x "%STT_WSL_REPO%/.venv/bin/python" >nul 2>nul && set "STT_RUNTIME=wsl"

REM ---- Banner (mirrors the GPU-fallback banner philosophy) ----
echo [stt-faster] runtime: %STT_RUNTIME%
echo [stt-faster] audio:   %STT_AUDIO_DIR_RESOLVED%
if /i "%STT_RUNTIME%"=="wsl" echo [stt-faster] WSL repo: %STT_WSL_REPO%

exit /b 0
