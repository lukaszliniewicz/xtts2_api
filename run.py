#!/usr/bin/env python3
"""XTTS FastAPI bootstrapper — detect hardware, install, start."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("run")

PROJECT_DIR = Path(__file__).parent.resolve()
SERVER_MODULE = "src.xtts_fastapi.main:app"

os.environ.setdefault("PIP_CACHE_DIR", str(PROJECT_DIR / ".pip-cache"))
os.environ.setdefault("PIXI_CACHE_DIR", str(PROJECT_DIR / ".pixi-cache"))
os.environ.setdefault("TMP", str(PROJECT_DIR / ".tmp"))
os.environ.setdefault("TEMP", str(PROJECT_DIR / ".tmp"))


def _find_pixi() -> str:
    exe = "pixi.exe" if platform.system() == "Windows" else "pixi"
    path = PROJECT_DIR / "bin" / exe
    if not path.is_file():
        log.error("pixi not found at %s", path)
        log.error("Run scripts/run.bat (Windows) or scripts/run.sh (Linux/macOS) first.")
        sys.exit(1)
    return str(path)


def _pip_install(*args: str) -> bool:
    pixi = _find_pixi()
    manifest = PROJECT_DIR / "pyproject.toml"
    cmd = [pixi, "run", "--manifest-path", str(manifest), "pip", "install"] + list(args)
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("pip install failed:\n%s", result.stderr.strip())
    return result.returncode == 0


def _check_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def detect_hardware(force_cpu: bool = False) -> str:
    if force_cpu:
        log.info("CPU forced via --cpu flag")
        return "cpu"

    system = platform.system()

    # NVIDIA CUDA via nvidia-smi
    try:
        r = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=15
        )
        if r.returncode == 0:
            log.info("Detected NVIDIA GPU (nvidia-smi)")
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # AMD ROCm on Linux
    if system == "Linux":
        try:
            r = subprocess.run(
                ["rocm-smi"], capture_output=True, text=True, timeout=15
            )
            if r.returncode == 0:
                log.info("Detected AMD GPU via ROCm (rocm-smi)")
                return "rocm"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        if Path("/opt/rocm").is_dir() and list(Path("/opt/rocm").glob("lib/librocm*")):
            log.info("Detected ROCm installation at /opt/rocm")
            return "rocm"

    # AMD ROCm on Windows (HIP SDK)
    if system == "Windows":
        hip_path = os.environ.get("HIP_PATH", "")
        if hip_path and Path(hip_path).is_dir():
            log.info("Detected AMD ROCm (HIP_PATH=%s)", hip_path)
            return "rocm"
        try:
            r = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True, text=True, timeout=15,
            )
            name = r.stdout.lower()
            if "amd" in name or "radeon" in name:
                log.info("Detected AMD GPU on Windows")
                return "rocm"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    log.info("No GPU accelerator detected, using CPU")
    return "cpu"


def check_backend(backend: str) -> bool:
    if not _check_package("torch"):
        return False
    if backend == "cpu":
        return True
    import torch
    return torch.cuda.is_available()


def ensure_torch(backend: str) -> bool:
    if backend == "cpu":
        pkgs = "torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0"
        idx = "https://download.pytorch.org/whl/cpu"
    elif backend == "cuda":
        pkgs = "torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0"
        idx = "https://download.pytorch.org/whl/cu126"
    elif backend == "rocm":
        pkgs = "torch==2.8.0"
        if platform.system() != "Windows":
            pkgs += " torchvision==0.23.0 torchaudio==2.8.0"
        idx = "https://download.pytorch.org/whl/rocm6.2"
    else:
        log.error("Unknown backend: %s", backend)
        return False

    if not _pip_install(*pkgs.split(), "--index-url", idx):
        return False

    import torch
    if backend in ("cuda", "rocm") and not torch.cuda.is_available():
        log.warning("torch installed but CUDA unavailable:")
        log.warning("  torch version: %s", torch.__version__)
        log.warning("  CUDA built: %s", torch.version.cuda)
        log.warning("Falling back to CPU torch")
        _pip_install("--force-reinstall", "--no-deps", "torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cpu")
        return True

    return True


def ensure_coqui_tts() -> bool:
    if _check_package("TTS"):
        return True
    log.info("Installing coqui-tts...")
    _pip_install("coqui-tts")
    if not _check_package("TTS"):
        log.warning("coqui-tts install had issues")
        return False
    _pip_install("transformers>=4.0,<5")
    return True


def ensure_deepspeed(backend: str) -> None:
    if backend != "cuda":
        return
    if _check_package("deepspeed"):
        return
    log.info("Installing deepspeed...")
    _pip_install("deepspeed==0.16.5")


def start_server() -> None:
    pixi = _find_pixi()
    manifest = PROJECT_DIR / "pyproject.toml"
    cmd = [
        pixi, "run", "--manifest-path", str(manifest),
        "python", "-m", "uvicorn",
        SERVER_MODULE,
        "--host", "0.0.0.0",
        "--port", "8020",
    ]
    log.info("Starting XTTS FastAPI server...\n")
    sys.stdout.flush()
    sys.stderr.flush()
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XTTS FastAPI server — auto-detect, install, start",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU even if GPU is available",
    )
    parser.add_argument(
        "--backend", choices=["auto", "cuda", "rocm", "cpu"], default="auto",
        help="Force a specific backend (default: auto-detect)",
    )
    args = parser.parse_args()

    backend = args.backend if args.backend != "auto" else detect_hardware(args.cpu)
    log.info("Selected backend: %s", backend)

    needs_install = not (check_backend(backend) and _check_package("TTS"))

    if needs_install:
        log.info("Dependencies missing or incomplete, installing...")
        if not ensure_torch(backend):
            log.error("Failed to install torch, aborting")
            sys.exit(1)
        if not ensure_coqui_tts():
            log.warning("coqui-tts installation had issues")
        ensure_deepspeed(backend)
        log.info("Installation complete.\n")
    else:
        log.info("All dependencies already installed.\n")

    start_server()


if __name__ == "__main__":
    main()
