@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\pythonw.exe" goto :launch

echo ======================================================================
echo Matte App environment was not found.
echo Starting one-click setup...
echo ======================================================================
echo.

call "%~dp0scripts\install.bat"

if not exist "%~dp0.venv\Scripts\pythonw.exe" (
    echo.
    echo [ERROR] Setup could not be completed.
    pause
    exit /b 1
)

:launch
if exist "%~dp0scripts\Run Matte App.vbs" (
    wscript.exe "%~dp0scripts\Run Matte App.vbs"
) else (
    start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0scripts\launch_app.pyw"
)
