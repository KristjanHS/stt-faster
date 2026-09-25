@echo off
setlocal
REM Build Transcribe-Setup.exe on the Windows host; no admin, no preinstalled tools. Output: <repo>\dist\ (make release attaches it).
REM Keep UV_VERSION equal to UV_VERSION in installer\setup_gui.py.
set "UV_VERSION=0.12.19"
set "PYINSTALLER_VERSION=6.22.3"
for %%I in ("%~dp0..") do set "REPO=%%~fI"
set "WORK=%TEMP%\stt-faster-build"

if exist "%WORK%" rmdir /s /q "%WORK%"
mkdir "%WORK%\uv" || ( echo Could not create %WORK% & pause & exit /b 1 )

echo Downloading uv %UV_VERSION% ...
curl -fsSL -o "%WORK%\uv.zip" "https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip" || ( echo uv download failed & pause & exit /b 1 )
tar -xf "%WORK%\uv.zip" -C "%WORK%\uv" || ( echo uv unzip failed & pause & exit /b 1 )
set "UV="
for /r "%WORK%\uv" %%F in (uv.exe) do if exist "%%F" set "UV=%%F"
if not defined UV ( echo uv.exe not found in the uv zip & pause & exit /b 1 )

REM Build from a local copy: PyInstaller cannot work from a \\wsl$ UNC path.
copy /y "%REPO%\installer\setup_gui.py" "%WORK%\setup_gui.py" >nul || ( echo Could not copy %REPO%\installer\setup_gui.py & pause & exit /b 1 )

echo Building Transcribe-Setup.exe with PyInstaller %PYINSTALLER_VERSION% ...
pushd "%WORK%"
"%UV%" tool run --python 3.12 --from "pyinstaller==%PYINSTALLER_VERSION%" pyinstaller --onefile --windowed --clean --noconfirm --name Transcribe-Setup setup_gui.py
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" ( echo PyInstaller failed with exit code %RC% & pause & exit /b 1 )

if not exist "%REPO%\dist" mkdir "%REPO%\dist"
copy /y "%WORK%\dist\Transcribe-Setup.exe" "%REPO%\dist\Transcribe-Setup.exe" >nul || ( echo Could not copy the exe into %REPO%\dist & pause & exit /b 1 )
echo Built %REPO%\dist\Transcribe-Setup.exe
pause
