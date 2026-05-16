from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8020
    models_dir: Path = Path("models")
    voices_dir: Path = Path("voices")
    files_dir: Path = Path("files")
    logs_dir: Path = Path("logs")
    coqui_tos_agreed: bool = True
    device: str = "auto"
    use_deepspeed: bool = True
    default_language: str = "en"
    default_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    default_model_local_dir: str = "XTTS_2.0.2"
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
    voice_cache_size: int = 100

    model_config = {"env_prefix": "xtts_", "env_file": ".env"}

    @model_validator(mode="after")
    def _apply_legacy_aliases(self):
        if self.max_ref_len is not None:
            self.max_ref_length = self.max_ref_len
        return self


settings = Settings()
