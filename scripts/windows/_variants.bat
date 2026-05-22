@echo off
REM _variants.bat - normalize VARIANTS (space-separated) -> VARIANTS_COMMA + VARIANT_COUNT.
REM
REM Caller contract:
REM   - `setlocal enabledelayedexpansion` before calling.
REM   - `set "VARIANTS=<space-separated ids>"` before calling.
REM   - Helper does NOT setlocal (outputs must flow back to caller).
REM
REM Outputs:
REM   VARIANTS_COMMA - comma-separated form ("1,36,44") for the --variants CLI flag
REM   VARIANT_COUNT  - integer count (used by _banner.bat to switch singular/plural)
REM
REM CMD-parser notes (mirrored across helpers):
REM   - Caller's `setlocal enabledelayedexpansion` propagates into this `call`,
REM     so `!VARIANTS!` / `!VARIANTS_COMMA!` resolve correctly here.

set "VARIANTS_COMMA="
set /a VARIANT_COUNT=0
for %%v in (!VARIANTS!) do (
    if "!VARIANTS_COMMA!"=="" (
        set "VARIANTS_COMMA=%%v"
    ) else (
        set "VARIANTS_COMMA=!VARIANTS_COMMA!,%%v"
    )
    set /a VARIANT_COUNT+=1
)

exit /b 0
