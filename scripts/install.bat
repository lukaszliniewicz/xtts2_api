@echo off
setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0.."
set "BIN_DIR=%PROJECT_DIR%\bin"
set "PIXI_EXE=%BIN_DIR%\pixi.exe"
set "FORCE="
set "BACKEND="
set "YES="

:parse_args
if not "%1"=="" (
    if /i "%1"=="--cpu" set BACKEND=cpu
    if /i "%1"=="--gpu" set BACKEND=cuda
    if /i "%1"=="--cuda" set BACKEND=cuda
    if /i "%1"=="--force" set FORCE=1
    if /i "%1"=="--yes" set YES=1
    shift
    goto parse_args
)

echo.
echo ============================================
echo  XTTS FastAPI Server - Installer
echo ============================================
echo.

:: Check if already installed
if exist "%PIXI_EXE%" (
    if not defined FORCE (
        echo pixi already found at %PIXI_EXE%
        echo Use --force to reinstall.
        goto :install_env
    )
)

:: Create bin directory
if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"

:: Download pixi binary
echo [1/4] Downloading pixi...
set "PIXI_VERSION=v0.68.1"
set "PIXI_URL=https://github.com/prefix-dev/pixi/releases/download/!PIXI_VERSION!/pixi-x86_64-pc-windows-msvc.exe"

powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PIXI_URL%' -OutFile '%PIXI_EXE%'" || (
    echo ERROR: Failed to download pixi from %PIXI_URL%
    exit /b 1
)

if not exist "%PIXI_EXE%" (
    echo ERROR: pixi binary not found after download
    exit /b 1
)

echo pixi downloaded to %PIXI_EXE%

:install_env
echo.
echo [2/4] Installing base environment...
cd /d "%PROJECT_DIR%"
"%PIXI_EXE%" install || (
    echo ERROR: pixi install failed
    exit /b 1
)

:: Choose backend
echo.
echo [3/4] Configuring PyTorch backend...

if not defined BACKEND (
    echo.
    set /p BACKEND="Select backend (cpu/cuda, default=cuda): "
    if "!BACKEND!"=="" set BACKEND=cuda
)

if /i "%BACKEND%"=="cuda" (
    echo Installing PyTorch with CUDA support...
    "%PIXI_EXE%" run install-torch-cuda || (
        echo WARNING: CUDA install failed, falling back to CPU...
        "%PIXI_EXE%" run install-torch-cpu
    )
) else (
    echo Installing PyTorch with CPU support...
    "%PIXI_EXE%" run install-torch-cpu
)

:: Install coqui-tts
echo.
echo [4/5] Installing coqui-tts...
"%PIXI_EXE%" run pip install coqui-tts || (
    echo WARNING: coqui-tts install had issues
)

:: Install deepspeed (CUDA only)
if /i "%BACKEND%"=="cuda" (
    echo.
    echo [5/5] Installing DeepSpeed...
    "%PIXI_EXE%" run install-deepspeed || (
        echo WARNING: DeepSpeed install had issues, continuing...
    )
)

:: Verify
echo.
echo ============================================
echo  Installation complete!
echo ============================================
echo.
"%PIXI_EXE%" run check-runtime
echo.
echo To start the server, run:
echo   scripts\run.bat
echo.
