@echo off
REM Audio Transcription - ESTONIAN, Desk-mic / close-talk (variant 44)
REM Edit the VARIANTS line below to use a different variant.
REM Space-separated for multiple variants (e.g. "1 36 44").
set "VARIANTS=44"
REM DIARIZE: "" = runtime default (wsl=on, docker=off); "0" = off; "1" = on
set "DIARIZE="
REM NUM_SPEAKERS: only used when diarize resolves to on; "" = auto-detect
set "NUM_SPEAKERS=2"

setlocal enabledelayedexpansion
set "STT_CALLER_DIR=%~dp0"
set "STT_TITLE=Audio Transcription - ESTONIAN (Desk-mic)"
set "STT_MODEL=TalTech Estonian Whisper (et-large)"
set "STT_LANG=Estonian"
set "STT_CLI_TAIL=--language et --output-format txt"
REM Helpers location. Default = WSL via UNC, works both in-place and when
REM copied next to audio. Edit if your repo lives elsewhere. Trailing \ required.
REM   set "STT_HELPERS_DIR=C:\projects\stt-faster\scripts\windows\"
REM   set "STT_HELPERS_DIR=D:\code\stt-faster\scripts\windows\"
REM   set "STT_HELPERS_DIR=\\wsl$\Ubuntu-22.04\home\you\stt-faster\scripts\windows\"
set "STT_HELPERS_DIR=\\wsl$\Ubuntu\home\kristjans\projects\stt-faster\scripts\windows\"

call "%STT_HELPERS_DIR%_runtime.bat" || ( pause & exit /b 1 )
call "%STT_HELPERS_DIR%_variants.bat"
call "%STT_HELPERS_DIR%_banner.bat"
call "%STT_HELPERS_DIR%_transcribe.bat"
call "%STT_HELPERS_DIR%_footer.bat"
