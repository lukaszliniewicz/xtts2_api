from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


def _default_models_dir() -> Path:
    """Use the Manager-provided stable storage root when XTTS has no override."""
    pandrator_models_dir = os.environ.get("PANDRATOR_MODELS_DIR")
    if pandrator_models_dir:
        return Path(pandrator_models_dir) / "xtts"
    return Path("models")


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8020
    models_dir: Path = Field(default_factory=_default_models_dir)
    voices_dir: Path = Path("voices")
    files_dir: Path = Path("files")
    logs_dir: Path = Path("logs")
    app_log_file: str = "app.log"
    access_log_file: str = "access.log"
    error_log_file: str = "errors.log"
    log_level: str = "INFO"
    log_max_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    log_backup_count: int = Field(default=5, ge=1)
    request_id_header: str = "X-Request-ID"
    coqui_tos_agreed: bool = True
    device: str = "auto"
    speech_backend: str = "xtts"
    voice_cloning_enabled: bool = True
    use_deepspeed: bool = True
    default_language: str = "en"
    default_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    default_model_local_dir: str = "XTTS_2.0.2"
    model_watch_enabled: bool = True
    model_watch_debounce_seconds: float = Field(default=1.0, ge=0.0)
    model_upload_chunk_size: int = Field(default=1024 * 1024, ge=64 * 1024, le=16 * 1024 * 1024)
    model_upload_max_file_bytes: int = Field(default=3 * 1024 * 1024 * 1024, ge=1)
    model_upload_max_total_bytes: int = Field(default=4 * 1024 * 1024 * 1024, ge=1)
    model_upload_max_config_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    gpt_cond_len: int = 12
    gpt_cond_chunk_len: int = 6
    max_ref_length: int = 12
    max_ref_len: int | None = Field(default=None, exclude=True)
    sound_norm_refs: bool = True
    librosa_trim_db: int | None = None
    min_ref_audio_seconds: float = 0.5
    temperature: float = 0.7
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 5.0
    length_penalty: float = 1.0
    enable_text_splitting: bool = False
    stream_chunk_size: int = 20
    overlap_wav_len: int = 1024
    voice_cache_size: int = Field(default=100, ge=0)

    model_config = {"env_prefix": "xtts_", "env_file": ".env"}

    @model_validator(mode="after")
    def _apply_legacy_aliases(self):
        if self.max_ref_len is not None:
            self.max_ref_length = self.max_ref_len
        return self


settings = Settings()
