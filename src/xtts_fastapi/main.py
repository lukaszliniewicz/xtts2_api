from __future__ import annotations

import base64
import ipaddress
import json
import logging
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from urllib.parse import unquote
from uuid import uuid4

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import model_loader
from .api_models import (
    CreateSpeechRequest,
    ModelDeletedResponse,
    FileDeletedResponse,
    FileListResponse,
    FileObject,
    ModelList,
    ModelUploadResponse,
    VoiceCreateResponse,
    VoiceList,
    XTTSParams,
)
from .audio import SUPPORTED_FORMATS
from .engine import engine
from .errors import APIError, unknown_model
from .file_store import file_store
from .logging_setup import (
    ACCESS_LOGGER_NAME,
    APP_LOGGER_NAME,
    ERROR_LOGGER_NAME,
    configure_file_logging,
    reset_request_id,
    set_request_id,
)
from .model_uploads import install_model_upload
from .model_lifecycle import delete_model as delete_installed_model
from .registry import normalize_model_id, registry
from .settings import settings
from .voices import normalize_voice_id, voice_store

configure_file_logging(
    logs_dir=settings.logs_dir,
    level=settings.log_level,
    max_bytes=settings.log_max_bytes,
    backup_count=settings.log_backup_count,
    app_log_file=settings.app_log_file,
    access_log_file=settings.access_log_file,
    error_log_file=settings.error_log_file,
)

app_logger = logging.getLogger(APP_LOGGER_NAME)
access_logger = logging.getLogger(ACCESS_LOGGER_NAME)
error_logger = logging.getLogger(ERROR_LOGGER_NAME)
INSTRUCTION_XTTS_FIELDS = set(XTTSParams.model_fields.keys())
MODEL_UPLOAD_MULTIPART_OVERHEAD_BYTES = 16 * 1024 * 1024

app = FastAPI(
    title="XTTS FastAPI Server",
    description="OpenAI-compatible text-to-speech server",
    version="0.1.3",
    docs_url="/",
)


def _sanitize_request_id(raw_request_id: str) -> str:
    allowed = {"-", "_", "."}
    sanitized = "".join(ch for ch in raw_request_id.strip() if ch.isalnum() or ch in allowed)
    return sanitized[:128]


def _model_upload_preflight_error(request: Request) -> JSONResponse | None:
    """Reject remote model mutations before parsing or filesystem work."""
    is_upload = request.method == "POST" and request.url.path == "/v1/models"
    is_delete = request.method == "DELETE" and request.url.path.startswith("/v1/models")
    if not is_upload and not is_delete:
        return None

    client_host = request.client.host if request.client is not None else ""
    try:
        client_address = ipaddress.ip_address(client_host.split("%", maxsplit=1)[0])
    except ValueError:
        client_address = None
    mapped = getattr(client_address, "ipv4_mapped", None)
    if client_address is None or not (client_address.is_loopback or (mapped and mapped.is_loopback)):
        return JSONResponse(
            {
                "error": {
                    "message": "Model uploads are restricted to loopback clients",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "model_upload_loopback_only",
                }
            },
            status_code=403,
        )

    if is_delete:
        raw_path = request.scope.get("raw_path", b"")
        if isinstance(raw_path, bytes):
            decoded_path = unquote(raw_path.decode("utf-8", errors="replace"))
        else:
            decoded_path = unquote(str(raw_path))
        model_prefix = "/v1/models/"
        if decoded_path.startswith(model_prefix):
            raw_model_id = decoded_path[len(model_prefix) :]
            if any(part in {".", ".."} for part in raw_model_id.split("/")):
                return JSONResponse(
                    {
                        "error": {
                            "message": "model_id contains a reserved, hidden, or unsafe path part",
                            "type": "invalid_request_error",
                            "param": "model_id",
                            "code": "invalid_model_id",
                        }
                    },
                    status_code=400,
                )
        return None

    raw_content_length = request.headers.get("content-length")
    try:
        content_length = int(raw_content_length or "")
    except ValueError:
        content_length = -1
    if content_length < 0:
        return JSONResponse(
            {
                "error": {
                    "message": "Model uploads require a valid Content-Length header",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "model_upload_length_required",
                }
            },
            status_code=411,
        )
    maximum_request_bytes = (
        settings.model_upload_max_total_bytes + MODEL_UPLOAD_MULTIPART_OVERHEAD_BYTES
    )
    if content_length > maximum_request_bytes:
        return JSONResponse(
            {
                "error": {
                    "message": f"Model upload request exceeds its limit of {maximum_request_bytes} bytes",
                    "type": "invalid_request_error",
                    "param": "files",
                    "code": "model_upload_too_large",
                }
            },
            status_code=413,
        )
    return None


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    header_name = settings.request_id_header or "X-Request-ID"
    incoming_request_id = request.headers.get(header_name, "")
    request_id = _sanitize_request_id(incoming_request_id) or uuid4().hex

    token = set_request_id(request_id)
    request.state.request_id = request_id
    started = time.perf_counter()
    response: Response | None = None

    try:
        response = _model_upload_preflight_error(request)
        if response is None:
            response = await call_next(request)
        return response
    finally:
        duration_ms = (time.perf_counter() - started) * 1000.0
        status = response.status_code if response is not None else 500
        client_host = request.client.host if request.client is not None else "-"
        content_length = "-"
        if response is not None:
            content_length = response.headers.get("content-length", "-")
            response.headers.setdefault(header_name, request_id)

        access_logger.info(
            "request_complete",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "duration_ms": f"{duration_ms:.2f}",
                "client": client_host,
                "content_length": content_length,
            },
        )
        reset_request_id(token)


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    record = {
        "method": request.method,
        "path": request.url.path,
        "status": exc.status,
        "code": exc.code or "invalid_request_error",
        "param": exc.param,
        "detail": exc.message,
    }

    if exc.status >= 500:
        error_logger.error("api_error", extra=record)
    else:
        app_logger.info("api_error", extra=record)

    return exc.to_response()


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    app_logger.info(
        "request_validation_error",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": 422,
            "detail": exc.errors(),
        },
    )
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    details = {
        "method": request.method,
        "path": request.url.path,
        "status": exc.status_code,
        "detail": exc.detail,
    }
    if exc.status_code >= 500:
        error_logger.error("http_exception", extra=details)
    else:
        app_logger.info("http_exception", extra=details)
    return await http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    error_logger.error(
        "unhandled_exception",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": 500,
            "detail": str(exc),
        },
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        {
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "param": None,
                "code": "internal_server_error",
            }
        },
        status_code=500,
    )


@app.on_event("startup")
async def startup():
    if settings.coqui_tos_agreed:
        import os

        os.environ["COQUI_TOS_AGREED"] = "1"
    registered = voice_store.register_staged_voices()
    if registered:
        app_logger.info("Registered %d staged voice(s) from %s", registered, settings.voices_dir)
    registry.discover()
    registry.start_watching()


@app.on_event("shutdown")
async def shutdown():
    registry.stop_watching()


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


def _ensure_voice_ingestion_supported() -> None:
    if settings.voice_cloning_enabled:
        return

    backend_name = (settings.speech_backend or "unknown").strip().lower() or "unknown"
    raise APIError(
        (
            f"Voice sample ingestion is not implemented for backend '{backend_name}'. "
            "This backend does not support voice cloning. "
            "Use GET /v1/audio/voices to list available voices."
        ),
        code="voice_ingestion_not_implemented",
        status=501,
    )


@app.get("/health")
async def health():
    payload = {
        "status": "ok",
        "version": "0.1.3",
        "model_count": len(registry.list_models()),
        "voice_count": len(voice_store.list_all()),
        "device": settings.device,
        "conditioning_cache": engine.conditioning_cache_info(),
    }
    if model_loader.HAS_XTTS:
        return payload

    payload["status"] = "unavailable"
    payload["runtime_error"] = str(model_loader.XTTS_IMPORT_ERROR or "unknown import error")
    return JSONResponse(payload, status_code=503)


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    models = registry.list_models(include_default=True)
    return ModelList(data=[m.to_openai() for m in models])


@app.post("/v1/models", response_model=ModelUploadResponse, status_code=201)
async def create_model(
    model_id: str | None = Form(default=None, description="Relative XTTS model identifier"),
    files: list[UploadFile] | None = File(default=None, description="XTTS model bundle files"),
):
    if model_id is None:
        raise APIError("model_id is required", param="model_id", code="missing_model_id")
    return await install_model_upload(model_id, files or [])


@app.delete("/v1/models", response_model=ModelDeletedResponse)
async def delete_models_root():
    return delete_installed_model("")


@app.delete("/v1/models/{model_id:path}", response_model=ModelDeletedResponse)
async def delete_model(model_id: str):
    return delete_installed_model(model_id)


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


@app.get("/v1/audio/voices", response_model=VoiceList)
@app.get("/v1/voices", response_model=VoiceList)
async def list_voices():
    return VoiceList(data=voice_store.list_all())


@app.post("/v1/audio/voices", response_model=VoiceCreateResponse)
async def create_voice(
    files: list[UploadFile] = File(..., description="Audio sample files"),
    voice_id: str = Form(None, description="Custom voice ID (normalized)"),
    model: str = Form(None, description="Associated model ID"),
    language: str = Form(None, description="Language code"),
):
    _ensure_voice_ingestion_supported()

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
    engine.clear_conditioning_cache()
    return result


@app.delete("/v1/audio/voices/{voice_id}")
@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str):
    try:
        deleted = voice_store.delete(voice_id)
    except ValueError as error:
        raise APIError(
            str(error), param="voice_id", code="invalid_voice_id", status=400
        ) from error
    if deleted:
        engine.clear_conditioning_cache()
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

    model_id = normalize_model_id(body.model)
    body = body.model_copy(update={"model": model_id})
    model_info = registry.get(model_id)
    supported_default_ids = {
        settings.default_model,
        "xtts_v2",
        "tts_models/multilingual/multi-dataset/xtts_v2",
    }
    if model_info is None and model_id not in supported_default_ids:
        raise unknown_model(model_id)

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
