from __future__ import annotations

import io
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000

SUPPORTED_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm"}

FFMPEG_ENCODERS: dict[str, tuple[str, str]] = {
    "mp3": ("libmp3lame", ".mp3"),
    "opus": ("libopus", ".opus"),
    "aac": ("aac", ".aac"),
    "flac": ("flac", ".flac"),
    "wav": ("pcm_s16le", ".wav"),
    "pcm": ("pcm_s16le", ".pcm"),
}


def has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def convert_wav_bytes(wav_bytes: bytes, target_format: str, sample_rate: int = SAMPLE_RATE) -> bytes:
    if target_format == "wav":
        return wav_bytes
    if target_format == "pcm":
        return _wav_to_pcm(wav_bytes)

    if not has_ffmpeg():
        logger.warning("ffmpeg not found, falling back to wav output")
        return wav_bytes

    encoder, ext = FFMPEG_ENCODERS[target_format]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
        f_in.write(wav_bytes)
        in_path = f_in.name
    out_path = Path(tempfile.mktemp(suffix=ext))
    try:
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-map_metadata", "-1",
            "-acodec", encoder,
            "-ar", str(sample_rate),
            "-ac", "1",
            out_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return out_path.read_bytes()
    finally:
        Path(in_path).unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


def _wav_to_pcm(wav_bytes: bytes) -> bytes:
    import struct

    if len(wav_bytes) < 44:
        return wav_bytes
    header_size = struct.unpack_from("<I", wav_bytes, 16)[0] + 20
    if header_size > len(wav_bytes):
        header_size = 44
    return wav_bytes[header_size:]


def numpy_to_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
