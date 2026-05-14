from __future__ import annotations

import base64
import json
import logging
import time
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .api_models import CreateSpeechRequest, ModelList, VoiceCreateResponse, VoiceList
from .audio import convert_wav_bytes, SUPPORTED_FORMATS
from .engine import engine
from .errors import APIError
from .registry import registry
from .settings import settings
from .voices import voice_store

logger = logging.getLogger(__name__)

app = FastAPI(
    title="XTTS FastAPI Server",
    description="OpenAI-compatible text-to-speech server",
    version="0.1.0",
    docs_url="/",
)


@app.exception_handler(APIError)
async def api_error_handler(request, exc: APIError):
    return exc.to_response()


@app.on_event("startup")
async def startup():
    if settings.coqui_tos_agreed:
        import os

        os.environ["COQUI_TOS_AGREED"] = "1"
    registry.discover()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "model_count": len(registry.list_models()),
        "voice_count": len(voice_store.list_all()),
        "device": settings.device,
    }


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    models = registry.list_models()
    if not models:
        return ModelList(data=[])
    return ModelList(data=[m.to_openai() for m in models])


@app.get("/v1/voices", response_model=VoiceList)
async def list_voices():
    return VoiceList(data=voice_store.list_all())


@app.post("/v1/voices", response_model=VoiceCreateResponse)
async def create_voice(
    files: list[UploadFile] = File(..., description="Audio sample files"),
    voice_id: str = Form(None, description="Custom voice ID"),
    model: str = Form(None, description="Associated model ID"),
    language: str = Form(None, description="Language code"),
):
    if not files:
        raise APIError("At least one audio file is required", param="files", code="missing_files")

    if voice_id is None:
        first_stem = files[0].filename
        if first_stem:
            voice_id = first_stem.rsplit(".", 1)[0]
        else:
            voice_id = f"voice_{int(time.time())}"

    file_data: list[tuple[str, bytes]] = []
    for f in files:
        content = await f.read()
        if not content:
            continue
        name = f.filename or f"sample_{len(file_data)}.wav"
        file_data.append((name, content))

    result = voice_store.create(voice_id, file_data, model=model, language=language)
    return result


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str):
    if voice_store.delete(voice_id):
        return {"deleted": True, "id": voice_id}
    raise APIError(f"Voice '{voice_id}' not found", param="voice_id", code="voice_not_found", status=404)


@app.post("/v1/audio/speech")
async def create_speech(body: CreateSpeechRequest):
    if body.response_format not in SUPPORTED_FORMATS:
        raise APIError(
            f"Unsupported response_format: {body.response_format}",
            param="response_format",
            code="unsupported_format",
        )

    try:
        model_info = registry.get(body.model)
    except Exception:
        model_info = None

    is_streaming = body.stream_format is not None

    if is_streaming:
        return await _stream_response(body, model_info)

    wav_bytes = await engine.generate_speech_async(body, model_info)
    content_type = f"audio/{body.response_format}"
    if body.response_format == "pcm":
        content_type = "audio/L16;rate=24000;channels=1"

    return Response(content=wav_bytes, media_type=content_type)


async def _stream_response(body: CreateSpeechRequest, model_info):
    if body.stream_format == "sse":
        return StreamingResponse(
            _sse_stream(body, model_info),
            media_type="text/event-stream",
        )
    else:
        return StreamingResponse(
            _audio_stream(body, model_info),
            media_type=f"audio/{body.response_format}",
        )


async def _audio_stream(body: CreateSpeechRequest, model_info) -> AsyncGenerator[bytes, None]:
    async for chunk in engine.generate_speech_stream_async(body, model_info):
        yield chunk


async def _sse_stream(body: CreateSpeechRequest, model_info) -> AsyncGenerator[str, None]:
    async for chunk in engine.generate_speech_stream_async(body, model_info):
        b64 = base64.b64encode(chunk).decode("ascii")
        event = {
            "type": "speech.audio.delta",
            "audio": b64,
        }
        yield f"data: {json.dumps(event)}\n\n"

    done_event = {
        "type": "speech.audio.done",
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }
    yield f"data: {json.dumps(done_event)}\n\n"


@app.post("/admin/models/refresh")
async def refresh_models():
    count = len(registry.refresh())
    return {"refreshed": count}
