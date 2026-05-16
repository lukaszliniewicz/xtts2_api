#!/usr/bin/env python3
"""XTTS FastAPI bootstrapper — detect hardware, install, start."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("run")

PROJECT_DIR = Path(__file__).parent.resolve()
SERVER_MODULE = "src.xtts_fastapi.main:app"

TORCH_VERSION = "2.6.0"
TORCHVISION_VERSION = "0.21.0"
TORCHAUDIO_VERSION = "2.6.0"
PIXI_PATH_OVERRIDE: Path | None = None

DEFAULT_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_MODEL_LOCAL_DIR = "XTTS_2.0.2"
DEFAULT_MODEL_REQUIRED_FILES = (
    "config.json",
    "model.pth",
    "speakers_xtts.pth",
    "vocab.json",
)
LEGACY_DEFAULT_MODEL_DIRS = ("v2.0.2",)

os.environ.setdefault("PIP_CACHE_DIR", str(PROJECT_DIR / ".pip-cache"))
os.environ.setdefault("PIXI_CACHE_DIR", str(PROJECT_DIR / ".pixi-cache"))
os.environ.setdefault("TMP", str(PROJECT_DIR / ".tmp"))
os.environ.setdefault("TEMP", str(PROJECT_DIR / ".tmp"))


def _find_pixi() -> str:
    if PIXI_PATH_OVERRIDE is not None:
        path = PIXI_PATH_OVERRIDE
    else:
        exe = "pixi.exe" if platform.system() == "Windows" else "pixi"
        path = PROJECT_DIR / "bin" / exe

    if not path.is_file():
        log.error("pixi not found at %s", path)
        if PIXI_PATH_OVERRIDE is not None:
            log.error("The value passed to --pixi-path must point to an existing pixi binary.")
        else:
            log.error("Run run.bat (Windows) or run.sh (Linux/macOS) first.")
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


def _default_model_path() -> Path:
    local_dir = os.environ.get("XTTS_DEFAULT_MODEL_LOCAL_DIR") or _read_dotenv_var(
        "XTTS_DEFAULT_MODEL_LOCAL_DIR"
    )
    return PROJECT_DIR / "models" / (local_dir or DEFAULT_MODEL_LOCAL_DIR)


def _default_model_id() -> str:
    return os.environ.get("XTTS_DEFAULT_MODEL") or _read_dotenv_var("XTTS_DEFAULT_MODEL") or DEFAULT_MODEL_ID


def _has_local_default_model() -> bool:
    model_dir = _default_model_path()
    if not model_dir.is_dir():
        return False
    return all((model_dir / name).is_file() for name in DEFAULT_MODEL_REQUIRED_FILES)


def _find_local_legacy_model() -> Path | None:
    for folder in LEGACY_DEFAULT_MODEL_DIRS:
        candidate = PROJECT_DIR / "models" / folder
        if candidate == _default_model_path():
            continue
        if candidate.is_dir() and all((candidate / name).is_file() for name in DEFAULT_MODEL_REQUIRED_FILES):
            return candidate
    return None


def _bool_value(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_dotenv_var(name: str) -> str | None:
    env_file = PROJECT_DIR / ".env"
    if not env_file.is_file():
        return None

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        return value.strip().strip("\"'")

    return None


def _is_coqui_tos_agreed() -> bool:
    checks = [
        os.environ.get("COQUI_TOS_AGREED"),
        os.environ.get("XTTS_COQUI_TOS_AGREED"),
        _read_dotenv_var("COQUI_TOS_AGREED"),
        _read_dotenv_var("XTTS_COQUI_TOS_AGREED"),
    ]
    return any(_bool_value(v) for v in checks)


def _resolve_default_hf_source(model_id: str) -> tuple[str, str] | None:
    normalized = model_id.strip().lower()
    if not normalized or "xtts" not in normalized:
        return None

    revision = "main"
    match = re.search(r"v\d+\.\d+\.\d+", normalized)
    if match is not None:
        revision = match.group(0)

    return "coqui/XTTS-v2", revision


def _move_model_dir(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))


def _download_default_model_from_hf(model_dir: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False

    source = _resolve_default_hf_source(_default_model_id())
    if source is None:
        return False

    repo_id, revision = source
    temp_dir = PROJECT_DIR / "models" / ".downloads" / f"{model_dir.name}-{uuid4().hex}"
    temp_dir.parent.mkdir(parents=True, exist_ok=True)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    try:
        log.info("Downloading default model directly from HuggingFace: %s (%s)", repo_id, revision)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(temp_dir),
            allow_patterns=[*DEFAULT_MODEL_REQUIRED_FILES, "hash.md5"],
        )
        missing = [name for name in DEFAULT_MODEL_REQUIRED_FILES if not (temp_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Missing required model files: {', '.join(missing)}")

        _move_model_dir(temp_dir, model_dir)
        return True
    except Exception as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        log.warning("Direct HuggingFace model download failed: %s", first_line)
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return False


def ensure_default_model() -> bool:
    if _has_local_default_model():
        return True

    model_dir = _default_model_path()
    legacy_model = _find_local_legacy_model()
    if legacy_model is not None:
        log.info("Copying default model from %s to %s", legacy_model, model_dir)
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        if model_dir.exists():
            shutil.rmtree(model_dir)
        shutil.copytree(legacy_model, model_dir)
        if _has_local_default_model():
            return True

    if not _is_coqui_tos_agreed():
        log.info(
            "Skipping default model predownload; set XTTS_COQUI_TOS_AGREED=true to download during install."
        )
        return False

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    os.environ["COQUI_TOS_AGREED"] = "1"

    if _download_default_model_from_hf(model_dir):
        if _has_local_default_model():
            log.info("Default model ready at %s", model_dir)
            return True

    if not _check_package("TTS"):
        log.warning("coqui-tts is not installed; skipping default model predownload")
        return False

    try:
        from TTS.utils.manage import ModelManager

        log.info("Ensuring default model in %s", model_dir)
        manager = ModelManager(output_prefix=str(PROJECT_DIR / "models"), progress_bar=True)
        downloaded_path, _config, _item = manager.download_model(_default_model_id())
        source_dir = downloaded_path if downloaded_path.is_dir() else downloaded_path.parent

        if source_dir.resolve() != model_dir.resolve():
            _move_model_dir(source_dir, model_dir)

        if _has_local_default_model():
            log.info("Default model ready at %s", model_dir)
            return True

        log.warning("Default model download completed but expected files are missing in %s", model_dir)
        return False
    except Exception as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        log.warning("Default model predownload failed: %s", first_line)
        return False


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
        pkgs = f"torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION}"
        idx = "https://download.pytorch.org/whl/cpu"
    elif backend == "cuda":
        pkgs = f"torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION}"
        idx = "https://download.pytorch.org/whl/cu126"
    elif backend == "rocm":
        pkgs = f"torch=={TORCH_VERSION}"
        if platform.system() != "Windows":
            pkgs += f" torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION}"
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
        _pip_install(
            "--force-reinstall",
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            f"torchaudio=={TORCHAUDIO_VERSION}",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        )
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
    _pip_install("--no-deps", "deepspeed==0.16.5")


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
    parser.add_argument(
        "--pixi-path",
        help="Path to an existing pixi binary (used instead of project-local bin/pixi)",
    )
    args = parser.parse_args()

    global PIXI_PATH_OVERRIDE
    if args.pixi_path:
        PIXI_PATH_OVERRIDE = Path(args.pixi_path).expanduser().resolve()
        log.info("Using pixi binary: %s", PIXI_PATH_OVERRIDE)

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

    ensure_default_model()

    start_server()


if __name__ == "__main__":
    main()
