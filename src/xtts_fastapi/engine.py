from __future__ import annotations

import asyncio
import logging
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
import torch

from .audio import convert_wav_bytes, numpy_to_wav, SAMPLE_RATE
from .errors import (
    invalid_reference_audio,
    missing_speaker_wav,
    reference_audio_too_short,
    unsupported_language,
)
from .file_store import file_store
from .model_loader import XTTSWrapper, XTTS_LANGUAGES, is_xtts_model
from .registry import ModelInfo
from .settings import settings
from .voices import voice_store

if TYPE_CHECKING:
    from .api_models import CreateSpeechRequest, XTTSParams

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self):
        self._models: dict[str, XTTSWrapper] = {}
        self._locks: dict[str, Lock] = {}
        self._latent_cache: dict[str, tuple] = {}

    def _get_lock(self, model_id: str) -> Lock:
        if model_id not in self._locks:
            self._locks[model_id] = Lock()
        return self._locks[model_id]

    def _get_wrapper(self, model_id: str, model_info: ModelInfo | None = None) -> XTTSWrapper:
        if model_id not in self._models:
            self._models[model_id] = XTTSWrapper(model_info)
        return self._models[model_id]

    def validate_language(self, language: str):
        lang = language.split("-")[0]
        if lang not in [l.split("-")[0] for l in XTTS_LANGUAGES]:
            raise unsupported_language(language)

    def _resolve_voice(self, request: CreateSpeechRequest) -> tuple[str, list[str] | None]:
        voice_id = request.voice_id

        if request.speaker_wav:
            resolved = []
            for item in request.speaker_wav:
                file_path = file_store.get_content_path(item)
                if file_path is not None:
                    resolved.append(str(file_path))
                else:
                    resolved.append(item)
            return voice_id, resolved

        if voice_id:
            meta = voice_store.get(voice_id)
            if meta:
                sample_paths = voice_store.get_sample_paths(voice_id)
                if sample_paths:
                    return voice_id, [str(p) for p in sample_paths]
                logger.warning("Voice '%s' has no samples, trying speaker lookup", voice_id)

            file_path = file_store.get_content_path(voice_id)
            if file_path is not None:
                return voice_id, [str(file_path)]

        return voice_id, None

    def _get_builtin_speaker(self, wrapper: XTTSWrapper, voice_id: str) -> tuple:
        sm = wrapper.speaker_manager
        if sm is not None and hasattr(sm, "speakers"):
            speakers = sm.speakers
            for name, data in speakers.items():
                if name.lower() == voice_id.lower():
                    gpt = data.get("gpt_conditioning_latents") or data.get("gpt_cond_latent")
                    spk = data.get("speaker_embedding")
                    if gpt is not None and spk is not None:
                        return (gpt, spk)
            for name, data in speakers.items():
                if voice_id.lower() in name.lower():
                    gpt = data.get("gpt_conditioning_latents") or data.get("gpt_cond_latent")
                    spk = data.get("speaker_embedding")
                    if gpt is not None and spk is not None:
                        return (gpt, spk)
        return None

    def _validate_reference_audio_paths(self, paths: list[str]) -> None:
        if not paths:
            return

        min_seconds = settings.min_ref_audio_seconds
        if min_seconds <= 0:
            return

        import soundfile as sf

        for path in paths:
            try:
                info = sf.info(path)
            except Exception as exc:
                raise invalid_reference_audio(path) from exc

            duration = 0.0
            if info.samplerate and info.samplerate > 0 and info.frames is not None:
                duration = float(info.frames) / float(info.samplerate)

            if duration < min_seconds:
                raise reference_audio_too_short(path, duration, min_seconds)

    def _build_inference_kwargs(self, xtts_params: XTTSParams | None, *, for_stream: bool) -> dict:
        kwargs = {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "top_k": settings.top_k,
            "repetition_penalty": settings.repetition_penalty,
            "length_penalty": settings.length_penalty,
            "enable_text_splitting": settings.enable_text_splitting,
        }

        if for_stream:
            kwargs["stream_chunk_size"] = settings.stream_chunk_size
            kwargs["overlap_wav_len"] = settings.overlap_wav_len

        if xtts_params is not None:
            pairs = [
                ("temperature", "temperature"),
                ("top_p", "top_p"),
                ("top_k", "top_k"),
                ("repetition_penalty", "repetition_penalty"),
                ("length_penalty", "length_penalty"),
                ("do_sample", "do_sample"),
                ("num_beams", "num_beams"),
                ("enable_text_splitting", "enable_text_splitting"),
            ]
            for attr, kw in pairs:
                val = getattr(xtts_params, attr)
                if val is not None:
                    kwargs[kw] = val

            if for_stream:
                stream_pairs = [
                    ("stream_chunk_size", "stream_chunk_size"),
                    ("overlap_wav_len", "overlap_wav_len"),
                ]
                for attr, kw in stream_pairs:
                    val = getattr(xtts_params, attr)
                    if val is not None:
                        kwargs[kw] = val

            if xtts_params.hf_generate_kwargs:
                kwargs.update(xtts_params.hf_generate_kwargs)

        return kwargs

    def _build_voice_kwargs(self, xtts_params: XTTSParams | None) -> dict:
        kwargs = {
            "gpt_cond_len": settings.gpt_cond_len,
            "gpt_cond_chunk_len": settings.gpt_cond_chunk_len,
            "max_ref_length": settings.max_ref_length,
            "sound_norm_refs": settings.sound_norm_refs,
        }

        if settings.librosa_trim_db is not None:
            kwargs["librosa_trim_db"] = settings.librosa_trim_db

        if xtts_params is not None:
            for attr, kw in [
                ("gpt_cond_len", "gpt_cond_len"),
                ("gpt_cond_chunk_len", "gpt_cond_chunk_len"),
                ("max_ref_length", "max_ref_length"),
                ("sound_norm_refs", "sound_norm_refs"),
                ("librosa_trim_db", "librosa_trim_db"),
            ]:
                val = getattr(xtts_params, attr)
                if val is not None:
                    kwargs[kw] = val

        return kwargs

    def generate_speech(self, request: CreateSpeechRequest, model_info: ModelInfo | None = None):
        self.validate_language(request.language)
        wrapper = self._get_wrapper(request.model, model_info)
        voice_id, speaker_wav_paths = self._resolve_voice(request)
        if speaker_wav_paths:
            self._validate_reference_audio_paths(speaker_wav_paths)
        infer_kwargs = self._build_inference_kwargs(request.xtts, for_stream=False)
        voice_kwargs = self._build_voice_kwargs(request.xtts)

        lock = self._get_lock(request.model)
        with lock:
            wrapper.load()

            gpt_cond_latent = None
            speaker_embedding = None

            if speaker_wav_paths:
                gpt_cond_latent, speaker_embedding = wrapper.get_conditioning_latents(
                    audio_path=speaker_wav_paths,
                    **voice_kwargs,
                )
            else:
                builtin = self._get_builtin_speaker(wrapper, voice_id)
                if builtin is not None:
                    gpt_cond_latent, speaker_embedding = builtin
                else:
                    raise missing_speaker_wav()

            if request.speed != 1.0:
                infer_kwargs["speed"] = request.speed

            result = wrapper.synthesize(
                text=request.input,
                language=request.language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                **infer_kwargs,
            )

        wav = np.array(result["wav"])
        wav_bytes = numpy_to_wav(wav, SAMPLE_RATE)

        if request.response_format != "wav":
            wav_bytes = convert_wav_bytes(wav_bytes, request.response_format)

        return wav_bytes

    def generate_speech_stream(self, request: CreateSpeechRequest, model_info: ModelInfo | None = None):
        self.validate_language(request.language)
        wrapper = self._get_wrapper(request.model, model_info)
        voice_id, speaker_wav_paths = self._resolve_voice(request)
        if speaker_wav_paths:
            self._validate_reference_audio_paths(speaker_wav_paths)
        infer_kwargs = self._build_inference_kwargs(request.xtts, for_stream=True)
        voice_kwargs = self._build_voice_kwargs(request.xtts)

        lock = self._get_lock(request.model)
        with lock:
            wrapper.load()

            gpt_cond_latent = None
            speaker_embedding = None

            if speaker_wav_paths:
                gpt_cond_latent, speaker_embedding = wrapper.get_conditioning_latents(
                    audio_path=speaker_wav_paths,
                    **voice_kwargs,
                )
            else:
                builtin = self._get_builtin_speaker(wrapper, voice_id)
                if builtin is not None:
                    gpt_cond_latent, speaker_embedding = builtin
                else:
                    raise missing_speaker_wav()

            if request.speed != 1.0:
                infer_kwargs["speed"] = request.speed

            for chunk in wrapper.synthesize_stream(
                text=request.input,
                language=request.language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                **infer_kwargs,
            ):
                wav_chunk = chunk.cpu().numpy() if torch.is_tensor(chunk) else np.array(chunk)
                wav_bytes = numpy_to_wav(wav_chunk, SAMPLE_RATE)

                if request.response_format != "wav":
                    wav_bytes = convert_wav_bytes(wav_bytes, request.response_format)

                yield wav_bytes

    async def generate_speech_async(self, request: CreateSpeechRequest, model_info: ModelInfo | None = None):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_speech, request, model_info)

    async def generate_speech_stream_async(self, request: CreateSpeechRequest, model_info: ModelInfo | None = None):
        loop = asyncio.get_event_loop()
        q = asyncio.Queue()

        def _run():
            try:
                for chunk in self.generate_speech_stream(request, model_info):
                    q.put_nowait(chunk)
            except Exception as e:
                q.put_nowait(e)
            finally:
                q.put_nowait(None)

        loop.run_in_executor(None, _run)

        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def refresh(self):
        self._models.clear()


engine = InferenceEngine()
