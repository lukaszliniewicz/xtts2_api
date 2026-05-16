from __future__ import annotations

import logging
import os
import platform
import inspect
import re
import sys
import gc
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4


def _nvcc_name() -> str:
    return "nvcc.exe" if platform.system() == "Windows" else "nvcc"


def _is_cuda_home(path: str | None) -> bool:
    if not path:
        return False
    root = Path(path)
    return (root / "bin" / _nvcc_name()).is_file()


def _configure_cuda_home_from_env() -> None:
    if _is_cuda_home(os.environ.get("CUDA_HOME")):
        return

    cuda_path = os.environ.get("CUDA_PATH")
    if _is_cuda_home(cuda_path):
        os.environ["CUDA_HOME"] = cuda_path  # type: ignore[index]
        return

    prefix = Path(sys.prefix)
    candidates: list[Path] = []
    if platform.system() == "Windows":
        candidates.append(prefix / "Library")
    else:
        candidates.append(prefix)

    for candidate in candidates:
        cuda_home = str(candidate)
        if not _is_cuda_home(cuda_home):
            continue

        os.environ["CUDA_HOME"] = cuda_home
        if platform.system() == "Windows":
            os.environ.setdefault("CUDA_PATH", cuda_home)

        bin_dir = str(candidate / "bin")
        current_path = os.environ.get("PATH", "")
        paths = current_path.split(os.pathsep) if current_path else []
        if all(Path(p).resolve() != Path(bin_dir).resolve() for p in paths if p):
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{current_path}" if current_path else bin_dir
        return


_configure_cuda_home_from_env()

import torch

from .settings import settings

if TYPE_CHECKING:
    from .registry import ModelInfo

logger = logging.getLogger(__name__)

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    HAS_XTTS = True
except ImportError:
    Xtts = None  # type: ignore
    XttsConfig = None  # type: ignore
    HAS_XTTS = False

try:
    from TTS.api import TTS as TTSSDK

    HAS_TTS_SDK = True
except ImportError:
    TTSSDK = None  # type: ignore
    HAS_TTS_SDK = False

try:
    from TTS.utils.manage import ModelManager

    HAS_MODEL_MANAGER = True
except ImportError:
    ModelManager = None  # type: ignore
    HAS_MODEL_MANAGER = False

try:
    from huggingface_hub import snapshot_download

    HAS_HF_HUB = True
except ImportError:
    snapshot_download = None  # type: ignore
    HAS_HF_HUB = False


XTTS_LANGUAGES = [
    "ar", "cs", "de", "en", "es", "fr", "hi", "hu",
    "it", "ja", "ko", "nl", "pl", "pt", "ru", "tr", "zh-cn",
]
LEGACY_DEFAULT_LOCAL_DIRS = ("v2.0.2",)
DEFAULT_MODEL_REQUIRED_FILES = (
    "config.json",
    "model.pth",
    "speakers_xtts.pth",
    "vocab.json",
)


def is_xtts_model(model_id: str) -> bool:
    return "xtts" in model_id.lower()


def _has_model_config(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _default_model_local_path() -> Path:
    return settings.models_dir / settings.default_model_local_dir


def _coqui_cache_root() -> Path:
    tts_home = os.environ.get("TTS_HOME")
    if tts_home:
        return Path(tts_home).expanduser()

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home).expanduser() / "tts"

    system = platform.system()
    if system == "Windows":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            return Path(local_appdata) / "tts"

        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata).expanduser().parent / "Local" / "tts"

        return Path.home() / "AppData" / "Local" / "tts"

    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "tts"

    return Path.home() / ".local" / "share" / "tts"


def _coqui_cache_folder_names(model_id: str) -> list[str]:
    raw = model_id.strip()
    if not raw:
        return []

    candidates = [raw]

    if "/" in raw:
        candidates.append("--".join(part for part in raw.split("/") if part))
    elif "xtts" in raw.lower():
        candidates.append(f"tts_models--multilingual--multi-dataset--{raw}")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)

    return deduped


def _find_cached_model_from_roots(model_name: str, roots: list[Path], label: str) -> Path | None:
    folder_names = _coqui_cache_folder_names(model_name)
    for root in roots:
        if not root.is_dir():
            continue

        for folder_name in folder_names:
            candidate = root / folder_name
            if _has_model_config(candidate):
                logger.info("Found default model in %s cache: %s", label, candidate)
                return candidate

    return None


def _hf_hub_roots() -> list[Path]:
    candidates: list[Path] = []

    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        candidates.append(Path(hf_hub_cache).expanduser())

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "hub")

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        candidates.append(Path(xdg_cache_home).expanduser() / "huggingface" / "hub")
    else:
        candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    roots: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        roots.append(path)
        seen.add(key)

    return roots


def _hf_repo_markers(model_name: str) -> set[str]:
    markers: set[str] = set()
    lowered = model_name.lower().strip()
    if not lowered:
        return markers

    tail = lowered.split("/")[-1]
    markers.add(tail)
    markers.add(tail.replace("_", "-"))

    if "xtts" in lowered:
        markers.add("xtts")

    return {item for item in markers if item}


def _find_hf_hub_snapshot(model_name: str) -> Path | None:
    markers = _hf_repo_markers(model_name)

    for hub_root in _hf_hub_roots():
        if not hub_root.is_dir():
            continue

        try:
            repos = [entry for entry in hub_root.iterdir() if entry.is_dir() and entry.name.startswith("models--")]
        except OSError:
            continue

        for repo_dir in repos:
            repo_name = repo_dir.name.lower()
            if markers and not any(marker in repo_name for marker in markers):
                continue

            snapshots_dir = repo_dir / "snapshots"
            if not snapshots_dir.is_dir():
                continue

            try:
                snapshots = sorted(
                    (entry for entry in snapshots_dir.iterdir() if entry.is_dir()),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
            except OSError:
                continue

            for snapshot in snapshots:
                if _has_model_config(snapshot):
                    logger.info("Found default model in HF hub cache: %s", snapshot)
                    return snapshot

    return None


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


def _download_default_model_from_hf(local: Path) -> bool:
    if not HAS_HF_HUB:
        return False

    source = _resolve_default_hf_source(settings.default_model)
    if source is None:
        return False

    repo_id, revision = source
    temp_dir = settings.models_dir / ".downloads" / f"{local.name}-{uuid4().hex}"
    temp_dir.parent.mkdir(parents=True, exist_ok=True)

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    try:
        logger.info("Downloading default model directly from HuggingFace: %s (%s)", repo_id, revision)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(temp_dir),
            allow_patterns=[*DEFAULT_MODEL_REQUIRED_FILES, "hash.md5"],
        )

        missing = [name for name in DEFAULT_MODEL_REQUIRED_FILES if not (temp_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Missing required model files: {', '.join(missing)}")

        _move_model_dir(temp_dir, local)
        return True
    except Exception as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        logger.warning("Direct HuggingFace model download failed: %s", first_line)
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return False


def _find_cached_default() -> Path | None:
    model_name = settings.default_model

    # Preferred local model path
    local = _default_model_local_path()
    if _has_model_config(local):
        logger.info("Found default model in local models/ directory")
        return local

    # Legacy local path from older releases
    legacy_local = settings.models_dir / model_name
    if _has_model_config(legacy_local):
        logger.info("Found default model in legacy local path")
        return legacy_local

    for legacy_folder in LEGACY_DEFAULT_LOCAL_DIRS:
        candidate = settings.models_dir / legacy_folder
        if _has_model_config(candidate):
            logger.info("Found default model in legacy local folder: %s", candidate)
            return candidate

    # Check coqui-tts cache roots
    cached = _find_cached_model_from_roots(
        model_name,
        [settings.models_dir / "tts", _coqui_cache_root()],
        label="coqui",
    )
    if cached is not None:
        return cached

    # Check HF_HOME legacy-style cache
    hf_home = Path(os.environ.get("HF_HOME", ""))
    if hf_home.is_dir():
        hf_cached = _find_cached_model_from_roots(model_name, [hf_home / "models"], label="HF_HOME")
        if hf_cached is not None:
            return hf_cached

    # Check HuggingFace Hub cache layout
    hf_snapshot = _find_hf_hub_snapshot(model_name)
    if hf_snapshot is not None:
        return hf_snapshot

    return None


def _copy_model_dir(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _download_default_model_to_local(local: Path) -> None:
    if settings.coqui_tos_agreed:
        os.environ["COQUI_TOS_AGREED"] = "1"

    settings.models_dir.mkdir(parents=True, exist_ok=True)

    if _download_default_model_from_hf(local):
        return

    if HAS_MODEL_MANAGER:
        logger.info("Downloading default model to %s", local)
        manager = ModelManager(output_prefix=str(settings.models_dir), progress_bar=True)
        downloaded_path, _config_path, _item = manager.download_model(settings.default_model)
        source_dir = downloaded_path if downloaded_path.is_dir() else downloaded_path.parent
        if source_dir.resolve() != local.resolve():
            logger.info("Moving downloaded default model into %s", local)
            _move_model_dir(source_dir, local)
        return

    if HAS_TTS_SDK:
        logger.info("Downloading default model from HuggingFace (one-time)...")
        TTSSDK(settings.default_model)
        return

    raise ImportError("coqui-tts model download utilities not available")


def _ensure_default_model_local() -> Path:
    local = _default_model_local_path()
    if _has_model_config(local):
        return local

    cached = _find_cached_default()
    if cached is not None and _has_model_config(cached):
        logger.info("Copying default model from cache to %s", local)
        try:
            _copy_model_dir(cached, local)
        except Exception as exc:
            logger.warning("Failed to copy cached default model to %s: %s", local, exc)
        else:
            if _has_model_config(local):
                return local

    _download_default_model_to_local(local)

    if _has_model_config(local):
        return local

    cached = _find_cached_default()
    if cached is not None and _has_model_config(cached):
        logger.info("Copying downloaded default model into %s", local)
        _copy_model_dir(cached, local)
        if _has_model_config(local):
            return local

    raise RuntimeError(
        f"Default model is unavailable at {local}. "
        "Set XTTS_COQUI_TOS_AGREED=true and verify internet access."
    )


def _resolve_device() -> str:
    dev = settings.device

    # If user explicitly chose a device, respect it (with fallback)
    if dev != "auto":
        if dev in ("cuda", "hip") and not torch.cuda.is_available():
            logger.warning("CUDA/ROCm requested but not available, falling back to CPU")
            return "cpu"
        if dev == "cpu":
            return "cpu"
        return dev

    # Auto-detect: CUDA/ROCm -> CPU
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(0).lower()
        if "amd" in cuda_name or "hip" in cuda_name or "radeon" in cuda_name:
            logger.info("Auto-detected AMD GPU via ROCm: %s", torch.cuda.get_device_name(0))
        else:
            logger.info("Auto-detected CUDA device: %s", torch.cuda.get_device_name(0))
        return "cuda"

    logger.warning("No GPU accelerator found, using CPU (this will be slow)")
    return "cpu"


class XTTSWrapper:
    def __init__(self, model_info: ModelInfo | None = None):
        self.device = _resolve_device()
        self.use_deepspeed = settings.use_deepspeed and self.device.startswith("cuda")
        self.model_info = model_info
        self.xtts_model = None
        self._speaker_manager = None
        self._loaded = False

    @property
    def model(self) -> Xtts:
        if self.xtts_model is None:
            raise RuntimeError("Model not loaded")
        return self.xtts_model

    @property
    def speaker_manager(self):
        return self._speaker_manager

    def load(self):
        if self._loaded and self.xtts_model is not None:
            return

        if not HAS_XTTS:
            raise ImportError("coqui-tts is not installed. Run the installer first.")

        if self.model_info is not None:
            self._load_from_folder()
        else:
            self._load_default()

        self._loaded = True
        logger.info("Model loaded on %s (deepspeed=%s)", self.device, self.use_deepspeed)

    def _load_from_folder(self):
        assert self.model_info is not None
        self._load_model_from_path(self.model_info.path)

    def _load_model_from_path(self, model_path: Path):
        config_path = model_path / "config.json"
        config = XttsConfig()
        config.load_json(str(config_path))
        model = Xtts.init_from_config(config)
        model = self._load_checkpoint_with_fallback(
            model=model,
            config=config,
            checkpoint_dir=str(model_path),
        )
        model.to(torch.device(self.device))
        self.xtts_model = model
        self._speaker_manager = getattr(model, "speaker_manager", None)

    def _load_default(self):
        model_path = _ensure_default_model_local()
        logger.info("Loading default model from %s", model_path)
        try:
            self._load_model_from_path(model_path)
        except RuntimeError as exc:
            if not (self.device.startswith("cuda") and self._is_cuda_runtime_error(exc)):
                raise
            logger.warning(
                "CUDA runtime error while loading default model (%s). Retrying on CPU.",
                str(exc).splitlines()[0],
            )
            self.use_deepspeed = False
            self.device = "cpu"
            self._load_model_from_path(model_path)

    def get_conditioning_latents(self, audio_path: list[str], **kwargs):
        self.load()
        supported_kwargs = self._supported_kwargs(self.model.get_conditioning_latents, kwargs)
        try:
            return self.model.get_conditioning_latents(audio_path=audio_path, **supported_kwargs)
        except RuntimeError as exc:
            if self._fallback_to_cpu(exc, "conditioning"):
                return self.model.get_conditioning_latents(audio_path=audio_path, **supported_kwargs)
            raise

    def synthesize(self, text: str, language: str, gpt_cond_latent, speaker_embedding, **kwargs):
        self.load()
        supported_kwargs = self._supported_kwargs(self.model.inference, kwargs)
        try:
            return self.model.inference(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                **supported_kwargs,
            )
        except RuntimeError as exc:
            if self._fallback_to_cpu(exc, "inference"):
                return self.model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=self._to_device(gpt_cond_latent),
                    speaker_embedding=self._to_device(speaker_embedding),
                    **supported_kwargs,
                )
            raise

    def synthesize_stream(self, text: str, language: str, gpt_cond_latent, speaker_embedding, **kwargs):
        self.load()
        supported_kwargs = self._supported_kwargs(self.model.inference_stream, kwargs)
        yielded = False
        try:
            for chunk in self.model.inference_stream(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                **supported_kwargs,
            ):
                yielded = True
                yield chunk
        except RuntimeError as exc:
            if yielded or not self._fallback_to_cpu(exc, "stream_inference"):
                raise
            for chunk in self.model.inference_stream(
                text=text,
                language=language,
                gpt_cond_latent=self._to_device(gpt_cond_latent),
                speaker_embedding=self._to_device(speaker_embedding),
                **supported_kwargs,
            ):
                yield chunk

    def _supported_kwargs(self, fn, kwargs: dict) -> dict:
        if not kwargs:
            return kwargs

        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return kwargs

        has_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )
        if has_var_kwargs:
            return kwargs

        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        dropped = sorted(k for k in kwargs.keys() if k not in allowed)
        if dropped:
            logger.debug("Dropping unsupported XTTS kwargs for %s: %s", fn.__name__, ", ".join(dropped))
        return filtered

    def _load_checkpoint_with_fallback(self, model, config, checkpoint_dir: str):
        use_deepspeed = self.use_deepspeed and self.device.startswith("cuda")

        try:
            model.load_checkpoint(
                config,
                checkpoint_dir=checkpoint_dir,
                use_deepspeed=use_deepspeed,
            )
            return model
        except Exception as exc:
            if not use_deepspeed:
                raise

            logger.warning(
                "DeepSpeed checkpoint load failed (%s). Retrying without DeepSpeed.",
                str(exc).splitlines()[0],
            )

            self.use_deepspeed = False
            model = Xtts.init_from_config(config)
            model.load_checkpoint(
                config,
                checkpoint_dir=checkpoint_dir,
                use_deepspeed=False,
            )
            return model

    def _fallback_to_cpu(self, exc: RuntimeError, stage: str) -> bool:
        if not self.device.startswith("cuda"):
            return False

        if not self._is_cuda_runtime_error(exc):
            return False

        logger.warning(
            "CUDA runtime error during %s (%s). Falling back to CPU for this model instance.",
            stage,
            str(exc).splitlines()[0],
        )

        self.use_deepspeed = False
        self.device = "cpu"
        self.xtts_model = None
        self._speaker_manager = None
        self._loaded = False

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        gc.collect()
        self.load()
        return True

    @staticmethod
    def _is_cuda_runtime_error(exc: RuntimeError) -> bool:
        msg = str(exc).lower()
        markers = [
            "cuda error",
            "device-side assert",
            "cublas",
            "cudnn",
            "illegal memory access",
        ]
        return any(marker in msg for marker in markers)

    def _to_device(self, value):
        if torch.is_tensor(value):
            return value.to(self.device)
        if isinstance(value, tuple):
            return tuple(self._to_device(v) for v in value)
        if isinstance(value, list):
            return [self._to_device(v) for v in value]
        if isinstance(value, dict):
            return {k: self._to_device(v) for k, v in value.items()}
        return value
