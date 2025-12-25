@echo off
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

