from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from .api_models import (
    CreateSpeechRequest,
    FileDeletedResponse,
    FileListResponse,
    FileObject,
    ModelList,
    XTTSParams,
    VoiceCreateResponse,
    VoiceList,
)
from .audio import convert_wav_bytes, SUPPORTED_FORMATS
from .engine import engine
from .errors import APIError
from .file_store import file_store
from .registry import registry
from .settings import settings
from .voices import normalize_voice_id, voice_store

logger = logging.getLogger(__name__)
INSTRUCTION_XTTS_FIELDS = set(XTTSParams.model_fields.keys())

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


def _parse_instruction_overrides(instructions: str | None) -> tuple[str | None, dict | None]:
    if instructions is None:
        return None, None

    raw = instructions.strip()
    if not raw or not raw.startswith("{"):
        return None, None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise APIError(
            "instructions must be valid JSON when used for XTTS overrides",
            param="instructions",
            code="invalid_instructions_json",
            status=422,
        ) from exc

    if not isinstance(payload, dict):
        raise APIError(
            "instructions JSON must decode to an object",
            param="instructions",
            code="invalid_instructions_json",
            status=422,
        )

    language_override = payload.get("language")
    if language_override is not None and not isinstance(language_override, str):
        raise APIError(
            "instructions.language must be a string",
            param="instructions",
            code="invalid_instructions_language",
            status=422,
        )

    xtts_overrides: dict[str, object] = {}

    raw_xtts = payload.get("xtts")
    if raw_xtts is not None:
        if not isinstance(raw_xtts, dict):
            raise APIError(
                "instructions.xtts must be a JSON object",
                param="instructions",
                code="invalid_instructions_xtts",
                status=422,
            )
        xtts_overrides.update(raw_xtts)

    for key in INSTRUCTION_XTTS_FIELDS:
        if key in payload:
            xtts_overrides[key] = payload[key]

    if "temp" in payload and "temperature" not in xtts_overrides:
        xtts_overrides["temperature"] = payload["temp"]

    if "max_ref_len" in payload and "max_ref_length" not in xtts_overrides:
        xtts_overrides["max_ref_length"] = payload["max_ref_len"]

    if "temp" in xtts_overrides:
        if "temperature" not in xtts_overrides:
            xtts_overrides["temperature"] = xtts_overrides["temp"]
        xtts_overrides.pop("temp")

    if "max_ref_len" in xtts_overrides:
        if "max_ref_length" not in xtts_overrides:
            xtts_overrides["max_ref_length"] = xtts_overrides["max_ref_len"]
        xtts_overrides.pop("max_ref_len")

    return language_override, xtts_overrides or None


def _apply_instruction_overrides(body: CreateSpeechRequest) -> CreateSpeechRequest:
    language_override, xtts_overrides = _parse_instruction_overrides(body.instructions)
    if language_override is None and xtts_overrides is None:
        return body

    payload = body.model_dump()

    default_language = CreateSpeechRequest.model_fields["language"].default
    if language_override is not None and body.language == default_language:
        payload["language"] = language_override

    if xtts_overrides is not None:
        existing_xtts = {}
        if body.xtts is not None:
            existing_xtts = body.xtts.model_dump(exclude_none=True)
        merged_xtts = {**xtts_overrides, **existing_xtts}
        try:
            payload["xtts"] = XTTSParams.model_validate(merged_xtts)
        except ValidationError as exc:
            details = exc.errors(include_url=False)
            message = details[0]["msg"] if details else str(exc)
            raise APIError(
                f"Invalid XTTS overrides in instructions: {message}",
                param="instructions",
                code="invalid_instructions_xtts",
                status=422,
            ) from exc

    return CreateSpeechRequest.model_validate(payload)


def _looks_like_wav(data: bytes) -> bool:
    if len(data) < 12:
        return False
    riff_header = data[:4]
    return riff_header in {b"RIFF", b"RIFX", b"RF64"} and data[8:12] == b"WAVE"


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


@app.post("/v1/files", response_model=VoiceCreateResponse)
async def create_file(
    file: UploadFile = File(..., description="File to upload"),
    purpose: str = Form(..., description="File purpose"),
    name: str | None = Form(default=None, description="Optional filename override"),
):
    filename = name or file.filename or "upload.bin"
    content = await file.read()

    if not purpose:
        raise APIError("purpose is required", param="purpose", code="missing_purpose")
    if not content:
        raise APIError("Uploaded file is empty", param="file", code="empty_file")
    if not _looks_like_wav(content):
        raise APIError(
            "Only WAV uploads are supported",
            param="file",
            code="unsupported_file_type",
            status=422,
        )

    raw_voice_id = Path(filename).stem
    voice_id = normalize_voice_id(raw_voice_id)
    if not voice_id:
        voice_id = f"voice-{int(time.time())}"
    normalized_sample_name = f"{voice_id}.wav"
    return voice_store.create(voice_id, [(normalized_sample_name, content)], model=None, language=None)


@app.get("/v1/files", response_model=FileListResponse)
async def list_files(
    limit: int = Query(default=100, ge=1, le=10_000),
    order: str = Query(default="desc", pattern=r"^(asc|desc)$"),
    purpose: str | None = Query(default=None),
    after: str | None = Query(default=None),
):
    return file_store.list_all(limit=limit, after=after, order=order, purpose=purpose)


@app.get("/v1/files/{file_id}", response_model=FileObject)
async def retrieve_file(file_id: str):
    file_obj = file_store.get(file_id)
    if file_obj is None:
        raise APIError(f"File '{file_id}' not found", param="file_id", code="file_not_found", status=404)
    return file_obj


@app.get("/v1/files/{file_id}/content")
async def retrieve_file_content(file_id: str):
    file_obj = file_store.get(file_id)
    if file_obj is None:
        raise APIError(f"File '{file_id}' not found", param="file_id", code="file_not_found", status=404)

    content = file_store.get_content(file_id)
    if content is None:
        raise APIError(f"File '{file_id}' content is missing", param="file_id", code="file_not_found", status=404)

    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{file_obj.filename}"'},
    )


@app.delete("/v1/files/{file_id}", response_model=FileDeletedResponse)
async def delete_file(file_id: str):
    if not file_store.delete(file_id):
        raise APIError(f"File '{file_id}' not found", param="file_id", code="file_not_found", status=404)
    return file_store.delete_response(file_id)


@app.get("/v1/voices", response_model=VoiceList)
async def list_voices():
    return VoiceList(data=voice_store.list_all())


@app.post("/v1/voices", response_model=VoiceCreateResponse)
async def create_voice(
    files: list[UploadFile] = File(..., description="Audio sample files"),
    voice_id: str = Form(None, description="Custom voice ID (normalized)"),
    model: str = Form(None, description="Associated model ID"),
    language: str = Form(None, description="Language code"),
):
    if not files:
        raise APIError("At least one audio file is required", param="files", code="missing_files")

    raw_voice_id = (voice_id or "").strip()
    if not raw_voice_id:
        first_name = files[0].filename or ""
        raw_voice_id = Path(first_name).stem if first_name else ""
    voice_id = normalize_voice_id(raw_voice_id)
    if not voice_id:
        voice_id = f"voice-{int(time.time())}"

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

    body = _apply_instruction_overrides(body)

    model_info = registry.get(body.model)
    supported_default_ids = {
        settings.default_model,
        "xtts_v2",
        "tts_models/multilingual/multi-dataset/xtts_v2",
    }
    if model_info is None and body.model not in supported_default_ids:
        raise APIError(
            f"Model '{body.model}' not found",
            param="model",
            code="model_not_found",
            status=404,
        )

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
