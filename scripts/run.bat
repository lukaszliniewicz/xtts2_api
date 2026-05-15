@echo off
setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0.."
set "BIN_DIR=%PROJECT_DIR%\bin"
set "PIXI_EXE=%BIN_DIR%\pixi.exe"

:: Download pixi if missing
if not exist "%PIXI_EXE%" (
    echo Downloading pixi...
    if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/prefix-dev/pixi/releases/download/v0.68.1/pixi-x86_64-pc-windows-msvc.exe' -OutFile '%PIXI_EXE%'"
    if errorlevel 1 (
        echo Failed to download pixi from GitHub.
        echo Check your internet connection or download manually.
        exit /b 1
    )
)

:: Ensure pixi env is installed
if not exist "%PROJECT_DIR%\.pixi\envs\default\python.exe" (
    echo Installing pixi environment...
    cd /d "%PROJECT_DIR%"
    "%PIXI_EXE%" install
    if errorlevel 1 (
        echo pixi install failed.
        exit /b 1
    )
) else (
    :: Ensure env is up to date with manifest
    cd /d "%PROJECT_DIR%"
    "%PIXI_EXE%" install --frozen >nul 2>&1 || "%PIXI_EXE%" install
)

:: Set local cache and temp dirs (E: has space, C: may not)
set "PIXI_CACHE_DIR=%PROJECT_DIR%\.pixi-cache"
set "PIP_CACHE_DIR=%PROJECT_DIR%\.pip-cache"
set "TMP=%PROJECT_DIR%\.tmp"
set "TEMP=%PROJECT_DIR%\.tmp"
if not exist "%PIXI_CACHE_DIR%" mkdir "%PIXI_CACHE_DIR%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
if not exist "%TMP%" mkdir "%TMP%"

:: Start bootstrapper
cd /d "%PROJECT_DIR%"
"%PIXI_EXE%" run python run.py %*
