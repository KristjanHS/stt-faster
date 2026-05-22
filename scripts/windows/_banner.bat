@echo off
REM _banner.bat - print the per-bat title header (5 content lines + separators).
REM
REM Caller contract:
REM   - `setlocal enabledelayedexpansion` before calling.
REM   - Caller sets: STT_TITLE, STT_MODEL, STT_LANG.
REM   - Optional: STT_TITLE_SUFFIX (e.g. "[DOCKER FORCED]" appended after title).
REM   - Helper reads VARIANTS_COMMA + VARIANT_COUNT (set earlier by _variants.bat).
REM   - Helper does NOT setlocal.
REM
REM Scratch vars (leak to caller scope - reserved name, do not reuse):
REM   STT_BANNER_TITLE - composed title line (STT_TITLE [+ optional suffix])
REM
REM CMD-parser notes:
REM   - Precompose the title string into STT_BANNER_TITLE so the echo line
REM     stays flat. A nested if/else with caret-escaped parens in the suffix
REM     would re-introduce the parser trap that bit the docker-forced bat.
REM   - The Variant/Variants singular/plural echo uses single-line `if (..) else (..)`
REM     with caret-escaped `^(`/`^)` - this pattern is safe ONLY when the
REM     if/else line is the deepest parens nesting in the executing context.
REM     A future caller wrapping this `call` in a parens block would push the
REM     line one level deeper and re-introduce the parser trap.

set "STT_BANNER_TITLE=!STT_TITLE!"
if defined STT_TITLE_SUFFIX if not "!STT_TITLE_SUFFIX!"=="" set "STT_BANNER_TITLE=!STT_TITLE! !STT_TITLE_SUFFIX!"

echo ========================================
echo !STT_BANNER_TITLE!
echo ========================================
echo Model:    !STT_MODEL!
echo Language: !STT_LANG!
if !VARIANT_COUNT!==1 (echo Variant:  !VARIANTS_COMMA!) else (echo Variants: !VARIANTS_COMMA! ^(!VARIANT_COUNT! variants^))
echo.

exit /b 0
