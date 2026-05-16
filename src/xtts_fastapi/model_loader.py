from __future__ import annotations

import logging
import os
import platform
import inspect
import sys
import gc
import shutil
from pathlib import Path
from typing import TYPE_CHECKING


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


XTTS_LANGUAGES = [
    "ar", "cs", "de", "en", "es", "fr", "hi", "hu",
    "it", "ja", "ko", "nl", "pl", "pt", "ru", "tr", "zh-cn",
]
LEGACY_DEFAULT_LOCAL_DIRS = ("v2.0.2",)


def is_xtts_model(model_id: str) -> bool:
    return "xtts" in model_id.lower()


def _has_model_config(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _default_model_local_path() -> Path:
    return settings.models_dir / settings.default_model_local_dir


def _coqui_cache_root() -> Path:
    system = platform.system()
    if system == "Windows":
        return Path(os.environ.get("APPDATA", "")) / "tts"
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "tts"
    return Path.home() / ".local" / "share" / "tts"


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

    # Check coqui-tts cache
    cached = _coqui_cache_root() / model_name
    if _has_model_config(cached):
        logger.info("Found default model in coqui cache")
        return cached

    # Check HF_HOME cache
    hf_home = Path(os.environ.get("HF_HOME", ""))
    if hf_home.is_dir():
        hf_cached = hf_home / "models" / model_name
        if _has_model_config(hf_cached):
            logger.info("Found default model in HF_HOME cache")
            return hf_cached

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

    if HAS_MODEL_MANAGER:
        logger.info("Downloading default model to %s", local)
        manager = ModelManager(output_prefix=str(settings.models_dir), progress_bar=True)
        downloaded_path, _config_path, _item = manager.download_model(settings.default_model)
        source_dir = downloaded_path if downloaded_path.is_dir() else downloaded_path.parent
        if source_dir.resolve() != local.resolve():
            logger.info("Copying downloaded default model into %s", local)
            _copy_model_dir(source_dir, local)
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
