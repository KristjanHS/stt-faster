@echo off
REM stt-faster - Docker setup. Builds the production image used as the fallback
REM runtime when WSL is unavailable. WSL users do not need to run this; the
REM transcribe bats auto-detect WSL and use it directly.

git pull
if errorlevel 1 echo WARNING: git pull failed. Continuing with the local checkout.

echo === stt-faster - Docker setup ===

where docker >nul 2>nul
if errorlevel 1 (
    echo ERROR: docker not found on PATH.
    echo Install Docker Desktop from https://docs.docker.com/desktop/install/windows-install/
    pause
    exit /b 1
)

docker info >nul 2>nul
if errorlevel 1 (
    echo ERROR: docker daemon is not reachable. Start Docker Desktop and re-run.
    pause
    exit /b 1
)

echo Building stt-faster:latest (this can take 5-15 minutes on first run)...
docker build -t stt-faster:latest "%~dp0."
if errorlevel 1 (
    echo ERROR: docker build failed.
    pause
    exit /b 1
)

REM Pre-create mount target dirs so docker run does not create them as root-owned.
if not exist "%USERPROFILE%\.cache\hf" mkdir "%USERPROFILE%\.cache\hf"
if not exist "%USERPROFILE%\.local\share\stt-faster" mkdir "%USERPROFILE%\.local\share\stt-faster"

echo.
echo === Setup complete ===
echo Next: double-click a transcribe_*.bat (or call it with STT_AUDIO_DIR set).
echo Models download lazily on first transcription run (~3 GB into %USERPROFILE%\.cache\hf).
pause
