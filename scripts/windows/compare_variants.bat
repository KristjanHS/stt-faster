@echo off
REM Audio Transcription - Builtin Variant List
REM Uses: New multi-variant runner with --variants option
REM Automatically uses variants from get_builtin_variants()
REM Processes a single audio file through all builtin variants
REM Outputs are organized in variant-specific subfolders

setlocal enabledelayedexpansion

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"

REM Convert Windows path to WSL path
set "WSL_INPUT_DIR=!SCRIPT_DIR:\=/!"
if "!SCRIPT_DIR:~0,1!"=="C" (
    set "WSL_INPUT_DIR=!WSL_INPUT_DIR:C:=/mnt/c!"
)
if "!SCRIPT_DIR:~0,1!"=="D" (
    set "WSL_INPUT_DIR=!WSL_INPUT_DIR:D:=/mnt/d!"
)
if "!SCRIPT_DIR:~0,1!"=="E" (
    set "WSL_INPUT_DIR=!WSL_INPUT_DIR:E:=/mnt/e!"
)

REM Find the audio file in the directory
for %%f in ("!SCRIPT_DIR!\*.wav" "!SCRIPT_DIR!\*.mp3" "!SCRIPT_DIR!\*.m4a" "!SCRIPT_DIR!\*.flac" "!SCRIPT_DIR!\*.ogg") do (
    set "AUDIO_FILE=%%f"
    goto :found
)

echo No audio file found in directory!
echo Supported formats: .wav, .mp3, .m4a, .flac, .ogg
pause
exit /b 1

:found
echo Audio file: !AUDIO_FILE!
echo.

REM Test WSL and Python access first
echo Testing WSL and Python access...
wsl -e bash -c "cd /home/kristjans/projects/stt-faster && .venv/bin/python --version" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Cannot access WSL or Python!
    echo ========================================
    echo.
    echo Attempting to show detailed error...
    wsl -e bash -c "cd /home/kristjans/projects/stt-faster && .venv/bin/python --version"
    echo.
    echo Press any key to exit...
    pause
    exit /b 1
)
echo WSL and Python access confirmed.
echo.

REM Get builtin variants from Python
echo Getting builtin variants...
echo.

REM Use a temp file to capture output and errors
set "TEMP_VARIANTS_FILE=%TEMP%\variants_output_%RANDOM%.txt"
wsl -e bash -c "cd /home/kristjans/projects/stt-faster && .venv/bin/python -c \"from backend.variants.registry import get_builtin_variants; variants = get_builtin_variants(); print(','.join(str(v.number) for v in variants))\"" > "!TEMP_VARIANTS_FILE!" 2>&1
set "PYTHON_ERROR=!errorlevel!"

if !PYTHON_ERROR! neq 0 (
    echo.
    echo ========================================
    echo ERROR: Failed to get builtin variants!
    echo ========================================
    echo.
    echo Python command failed with error code !PYTHON_ERROR!
    echo.
    echo Error output:
    type "!TEMP_VARIANTS_FILE!"
    echo.
    del "!TEMP_VARIANTS_FILE!" >nul 2>&1
    echo Press any key to exit...
    pause
    exit /b !PYTHON_ERROR!
)

REM Read the variants from the temp file
set "VARIANTS="
for /f "delims=" %%i in (!TEMP_VARIANTS_FILE!) do set "VARIANTS=%%i"
    del "!TEMP_VARIANTS_FILE!" >nul 2>&1

if "!VARIANTS!"=="" (
    echo.
    echo ========================================
    echo ERROR: No variants returned!
    echo ========================================
    echo.
    echo The Python command succeeded but returned no variant numbers.
    echo.
    echo Press any key to exit...
    pause
    exit /b 1
)

echo.
echo ========================================
echo Builtin Variant Comparison
echo ========================================
echo.
echo Model: TalTech Estonian Whisper (et-large)
echo Language: Estonian
echo Variants: !VARIANTS! (from get_builtin_variants)
echo Processing audio file: !AUDIO_FILE!
echo WSL path: !WSL_INPUT_DIR!
echo.

REM Run the transcription script via WSL with custom variant list
REM Environment variables:
REM   HF_HOME - Hugging Face cache location
REM   HF_HUB_CACHE - Hugging Face hub cache
REM Output: Both txt and json formats
REM Output structure: variant_outputs/variant_XXX_name/processed/
REM Note: transcribe_manager.py processes folders, so we pass the directory containing the file
echo Starting transcription...
echo NOTE: You will see real-time progress for each variant as it processes.
echo.

REM Run transcription with output going directly to console for real-time display
wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && cd /home/kristjans/projects/stt-faster && .venv/bin/python scripts/transcribe_manager.py process '!WSL_INPUT_DIR!' --preset et-large --language et --output-format both --variants '!VARIANTS!'"
set "TRANSCRIBE_ERROR=!errorlevel!"

echo.
if !TRANSCRIBE_ERROR! neq 0 (
    echo ========================================
    echo ERROR: Transcription failed with error code !TRANSCRIBE_ERROR!
    echo ========================================
    echo.
    echo The transcription process encountered an error.
    echo Check the output above for detailed error information.
    echo.
    echo Press any key to exit...
    pause
    exit /b !TRANSCRIBE_ERROR!
)

echo ========================================
echo Processing Complete!
echo ========================================
echo.
echo Outputs are organized in: %~dp0variant_outputs\
echo Each variant has its own folder with processed/ and run_meta.json
echo.

echo.
echo ========================================
echo Generating variant report...
echo ========================================
echo.
wsl -e bash -c "cd /home/kristjans/projects/stt-faster && make variant-report"
set "REPORT_ERROR=!errorlevel!"

if !REPORT_ERROR! neq 0 (
    echo.
    echo ========================================
    echo WARNING: Variant report generation failed with error code !REPORT_ERROR!
    echo ========================================
    echo.
    echo Check the output above for error details.
    echo.
) else (
    echo.
    echo Variant report generated successfully.
    echo.
)

pause

