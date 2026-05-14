@echo off
setlocal

set "PROJECT_DIR=%~dp0.."
set "PIXI_ENV=%PROJECT_DIR%\.pixi\envs\default"

:: CUDA_HOME required by deepspeed on Windows
set "CUDA_HOME=%PIXI_ENV%\Library"
set "CUDA_PATH=%PIXI_ENV%\Library"
set "CUDA_TOOLKIT_ROOT_DIR=%PIXI_ENV%\Library"

:: Launch server via uvicorn
"%PIXI_ENV%\python.exe" -m uvicorn src.xtts_fastapi.main:app --host 0.0.0.0 --port 8020
