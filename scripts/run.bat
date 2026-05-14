@echo off
setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0.."
set "BIN_DIR=%PROJECT_DIR%\bin"
set "PIXI_EXE=%BIN_DIR%\pixi.exe"
set "PIXI_ENV=%PROJECT_DIR%\.pixi\envs\default"
set "INSTALL_SCRIPT=%~dp0install.bat"

:: CUDA_HOME required by deepspeed on Windows
set "CUDA_HOME=%PIXI_ENV%\Library"
set "CUDA_PATH=%PIXI_ENV%\Library"
set "CUDA_TOOLKIT_ROOT_DIR=%PIXI_ENV%\Library"

:: Check if pixi exists, if not run installer
if not exist "%PIXI_EXE%" (
    echo pixi not found. Running installer first...
    call "%INSTALL_SCRIPT%" --yes
    if errorlevel 1 (
        echo Installer failed. Please run scripts\install.bat manually.
        exit /b 1
    )
)

:: Start server
echo Starting XTTS FastAPI server...
cd /d "%PROJECT_DIR%"
"%PIXI_EXE%" run serve
