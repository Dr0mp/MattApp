@echo off
setlocal enabledelayedexpansion
title Matte App - Setup and Portable Environment Installer
cd /d "%~dp0.."
set "ROOT=%CD%"

echo ======================================================================
echo             Matte App - One-Click Portable Setup
echo ======================================================================
echo.

:: 1. Locate or download portable uv
set "UV_EXE=%ROOT%\bin\uv.exe"
if exist "%UV_EXE%" (
    echo [OK] Using portable uv at bin\uv.exe
) else (
    where uv >nul 2>&1
    if !errorlevel! equ 0 (
        set "UV_EXE=uv"
        echo [OK] Using uv from system PATH
    ) else (
        echo [INFO] Portable uv not found. Downloading standalone uv for Windows...
        if not exist "%ROOT%\bin" mkdir "%ROOT%\bin"
        powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile '%ROOT%\bin\uv.zip'; Expand-Archive -Path '%ROOT%\bin\uv.zip' -DestinationPath '%ROOT%\bin' -Force; Remove-Item '%ROOT%\bin\uv.zip' -Force"
        if not exist "%ROOT%\bin\uv.exe" (
            echo [ERROR] Failed to download portable uv. Please check your internet connection.
            pause
            exit /b 1
        )
        set "UV_EXE=%ROOT%\bin\uv.exe"
        echo [OK] Portable uv successfully installed to bin\uv.exe
    )
)

:: Add bin to PATH so child scripts can access uv
set "PATH=%ROOT%\bin;!PATH!"

echo.
echo [1/3] Creating virtual environment (.venv) with Python 3.12...
"%UV_EXE%" venv --python 3.12 "%ROOT%\.venv"
if !errorlevel! neq 0 (
    echo [ERROR] Failed to create .venv virtual environment.
    pause
    exit /b 1
)

echo.
echo [2/3] Installing PyTorch with CUDA 12.6 into .venv...
"%UV_EXE%" pip install --python "%ROOT%\.venv\Scripts\python.exe" torch torchvision --index-url https://download.pytorch.org/whl/cu126
if !errorlevel! neq 0 (
    echo [WARNING] CUDA PyTorch install returned non-zero. Retrying default PyTorch...
    "%UV_EXE%" pip install --python "%ROOT%\.venv\Scripts\python.exe" torch torchvision
)

echo.
echo [3/3] Installing Matte App and dependencies (PySide6, huggingface-hub, BiRefNet)...
"%UV_EXE%" pip install --python "%ROOT%\.venv\Scripts\python.exe" -e ".[birefnet]"
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install application dependencies.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo [SUCCESS] Matte App environment is ready!
echo Launch the app anytime by double-clicking "Run Matte App.bat"
echo ======================================================================
echo.
if "%~1"=="" pause
