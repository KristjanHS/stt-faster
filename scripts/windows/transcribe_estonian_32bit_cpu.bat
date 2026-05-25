@echo off
REM Audio Transcription - ESTONIAN, CPU-only (et-32 preset, variant 7)
REM Variant 7: no preprocessing + minimal transcription parameters (no VAD filter).
REM Edit the VARIANTS line below to use a different variant.
set "VARIANTS=7"
REM DIARIZE: "" = runtime default (wsl=on, docker=off); "0" = off; "1" = on
set "DIARIZE="
REM NUM_SPEAKERS: only used when diarize resolves to on; "" = auto-detect
set "NUM_SPEAKERS=2"

setlocal enabledelayedexpansion
set "STT_CALLER_DIR=%~dp0"
set "STT_TITLE=Audio Transcription - ESTONIAN (32-bit CPU)"
set "STT_MODEL=TalTech Estonian Whisper (et-32, CPU-optimized)"
set "STT_LANG=Estonian"
set "STT_CLI_TAIL=--preset et-32 --language et --output-format txt"
REM Helpers location. Default = WSL via UNC, works both in-place and when
REM copied next to audio. Edit if your repo lives elsewhere. Trailing \ required.
REM   set "STT_HELPERS_DIR=C:\projects\stt-faster\scripts\windows\"
REM   set "STT_HELPERS_DIR=D:\code\stt-faster\scripts\windows\"
REM   set "STT_HELPERS_DIR=\\wsl$\Ubuntu-22.04\home\you\stt-faster\scripts\windows\"
set "STT_HELPERS_DIR=\\wsl$\Ubuntu-22.04\home\kristjans\projects\stt-faster\scripts\windows\"

call "%STT_HELPERS_DIR%_runtime.bat" || ( pause & exit /b 1 )
call "%STT_HELPERS_DIR%_variants.bat"
call "%STT_HELPERS_DIR%_banner.bat"
call "%STT_HELPERS_DIR%_transcribe.bat"
call "%STT_HELPERS_DIR%_footer.bat"
