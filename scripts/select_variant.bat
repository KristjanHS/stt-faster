@echo off
REM Emit the active uv extras variant ("cpu" or "cu130") for this checkout.
REM
REM Reads <repo>\.stt-variant.local (gitignored). Absent file => "cpu". Anything
REM other than "cpu" or "cu130" is rejected with a clear error so a typo doesn't
REM silently fall back to the default.
REM
REM See docs/plans/archived/2026-05-21-cuda-deps-cpu-gpu-extras-design.md.

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "VARIANT_FILE=%SCRIPT_DIR%..\.stt-variant.local"
set "VARIANT=cpu"

if exist "%VARIANT_FILE%" (
    for /f "usebackq tokens=1 delims= 	" %%a in ("%VARIANT_FILE%") do (
        set "VARIANT=%%a"
        goto :validate
    )
)

:validate
if /i "!VARIANT!"=="cpu" (
    echo cpu
    endlocal & exit /b 0
)
if /i "!VARIANT!"=="cu130" (
    echo cu130
    endlocal & exit /b 0
)

echo unknown variant '!VARIANT!' in .stt-variant.local -- expected 'cpu' or 'cu130' 1>&2
endlocal & exit /b 1
