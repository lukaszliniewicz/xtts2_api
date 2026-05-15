from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING

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


XTTS_LANGUAGES = [
    "ar", "cs", "de", "en", "es", "fr", "hi", "hu",
    "it", "ja", "ko", "nl", "pl", "pt", "ru", "tr", "zh-cn",
]


def is_xtts_model(model_id: str) -> bool:
    return "xtts" in model_id.lower()


def _coqui_cache_root() -> Path:
    system = platform.system()
    if system == "Windows":
        return Path(os.environ.get("APPDATA", "")) / "tts"
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "tts"
    return Path.home() / ".local" / "share" / "tts"


def _find_cached_default() -> Path | None:
    model_name = settings.default_model

    # Check local models/ directory first
    local = settings.models_dir / model_name
    if local.is_dir() and (local / "config.json").is_file():
        logger.info("Found default model in local models/ directory")
        return local

    # Check coqui-tts cache
    cached = _coqui_cache_root() / model_name
    if cached.is_dir() and (cached / "config.json").is_file():
        logger.info("Found default model in coqui cache")
        return cached

    # Check HF_HOME cache
    hf_home = Path(os.environ.get("HF_HOME", ""))
    if hf_home.is_dir():
        hf_cached = hf_home / "models" / model_name
        if hf_cached.is_dir() and (hf_cached / "config.json").is_file():
            logger.info("Found default model in HF_HOME cache")
            return hf_cached

    return None


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
        model_path = self.model_info.path
        config_path = model_path / "config.json"
        config = XttsConfig()
        config.load_json(str(config_path))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=str(model_path),
            use_deepspeed=self.use_deepspeed,
        )
        model.to(torch.device(self.device))
        self.xtts_model = model
        self._speaker_manager = getattr(model, "speaker_manager", None)

    def _load_default(self):
        if not HAS_TTS_SDK:
            raise ImportError("coqui-tts SDK not available")

        # Try loading from local cache first, avoiding HF download
        cached_path = _find_cached_default()
        if cached_path is not None:
            logger.info("Loading default model from %s", cached_path)
            config = XttsConfig()
            config.load_json(str(cached_path / "config.json"))
            model = Xtts.init_from_config(config)
            model.load_checkpoint(config, checkpoint_dir=str(cached_path), use_deepspeed=self.use_deepspeed)
            model.to(torch.device(self.device))
            self.xtts_model = model
            self._speaker_manager = getattr(model, "speaker_manager", None)
            return

        if settings.coqui_tos_agreed:
            os.environ["COQUI_TOS_AGREED"] = "1"
        logger.info("Downloading default model from HuggingFace (one-time)...")
        tts = TTSSDK(settings.default_model).to(self.device)
        self.xtts_model = tts.synthesizer.tts_model
        self._speaker_manager = getattr(self.xtts_model, "speaker_manager", None)

    def get_conditioning_latents(self, audio_path: list[str], **kwargs):
        self.load()
        return self.model.get_conditioning_latents(audio_path=audio_path, **kwargs)

    def synthesize(self, text: str, language: str, gpt_cond_latent, speaker_embedding, **kwargs):
        self.load()
        return self.model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            **kwargs,
        )

    def synthesize_stream(self, text: str, language: str, gpt_cond_latent, speaker_embedding, **kwargs):
        self.load()
        return self.model.inference_stream(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            **kwargs,
        )
