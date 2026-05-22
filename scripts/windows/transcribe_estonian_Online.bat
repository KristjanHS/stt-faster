@echo off
REM Audio Transcription - ESTONIAN, Online / mixed-source (variant 52)
REM Edit the VARIANTS line below to use a different variant.
REM Space-separated for multiple variants (e.g. "1 36 44").
set "VARIANTS=52"
set "DIARIZE="          REM "" = runtime default (wsl=on, docker=off); "0" = off; "1" = on
set "NUM_SPEAKERS=2"    REM only used when diarize resolves to on; "" = auto-detect

setlocal enabledelayedexpansion
set "STT_CALLER_DIR=%~dp0"
set "STT_TITLE=Audio Transcription - ESTONIAN (Online)"
set "STT_MODEL=TalTech Estonian Whisper (et-large)"
set "STT_LANG=Estonian"
set "STT_CLI_TAIL=--language et --output-format txt"
call "%~dp0_runtime.bat" || ( pause & exit /b 1 )
call "%~dp0_variants.bat"
call "%~dp0_banner.bat"
call "%~dp0_transcribe.bat"
call "%~dp0_footer.bat"
