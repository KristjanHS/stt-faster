@echo off
REM _footer.bat - print "Processing Complete!" trailer + pause.
REM
REM Caller contract:
REM   - `setlocal enabledelayedexpansion` before calling.
REM   - Caller (or _runtime.bat) populated STT_AUDIO_DIR_RESOLVED.
REM   - Helper does NOT setlocal.

echo.
echo ========================================
echo Processing Complete!
echo ========================================
echo Processed: !STT_AUDIO_DIR_RESOLVED!\processed\
echo Failed:    !STT_AUDIO_DIR_RESOLVED!\failed\
pause

exit /b 0
