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
REM CMD-parser notes:
REM   - Precompose the title string into STT_BANNER_TITLE so the echo line
REM     stays flat. A nested if/else with caret-escaped parens in the suffix
REM     would re-introduce the parser trap that bit the docker-forced bat.
REM   - The Variant/Variants singular/plural echo uses single-line `if (..) else (..)`
REM     with caret-escaped `^(`/`^)` - this is the legacy pattern from the
REM     existing bats and is safe at a single nesting level.

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
