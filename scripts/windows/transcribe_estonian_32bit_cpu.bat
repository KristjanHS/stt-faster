@echo off
REM Audio Transcription - ESTONIAN, CPU-only (et-32 preset, variant 7)
REM Variant 7: no preprocessing + minimal transcription parameters (no VAD filter).
REM Edit the VARIANTS line below to use a different variant.
set "VARIANTS=7"
set "DIARIZE="          REM "" = runtime default (wsl=on, docker=off); "0" = off; "1" = on
set "NUM_SPEAKERS=2"    REM only used when diarize resolves to on; "" = auto-detect

setlocal enabledelayedexpansion
set "STT_CALLER_DIR=%~dp0"
set "STT_TITLE=Audio Transcription - ESTONIAN (32-bit CPU)"
set "STT_MODEL=TalTech Estonian Whisper (et-32, CPU-optimized)"
set "STT_LANG=Estonian"
set "STT_CLI_TAIL=--preset et-32 --language et --output-format txt"
call "%~dp0_runtime.bat" || ( pause & exit /b 1 )
call "%~dp0_variants.bat"
call "%~dp0_banner.bat"
call "%~dp0_transcribe.bat"
call "%~dp0_footer.bat"
