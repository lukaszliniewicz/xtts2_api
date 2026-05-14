from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8020
    models_dir: Path = Path("models")
    voices_dir: Path = Path("voices")
    coqui_tos_agreed: bool = False
    device: str = "cuda"
    use_deepspeed: bool = True
    default_language: str = "en"
    default_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    gpt_cond_len: int = 12
    gpt_cond_chunk_len: int = 4
    max_ref_len: int = 10
    sound_norm_refs: bool = False
    temperature: float = 0.85
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 2.0
    length_penalty: float = 1.0
    stream_chunk_size: int = 20
    overlap_wav_len: int = 1024
    voice_cache_size: int = 100

    model_config = {"env_prefix": "xtts_", "env_file": ".env"}


settings = Settings()
