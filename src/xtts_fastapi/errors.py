from __future__ import annotations

from fastapi.responses import JSONResponse


class APIError(Exception):
    def __init__(self, message: str, param: str | None = None, code: str | None = None, status: int = 400):
        self.message = message
        self.param = param
        self.code = code
        self.status = status

    def to_response(self) -> JSONResponse:
        body = {
            "error": {
                "message": self.message,
                "type": "invalid_request_error",
                "param": self.param,
                "code": self.code or "invalid_request_error",
            }
        }
        return JSONResponse(body, status_code=self.status)


def unknown_model(model_id: str) -> APIError:
    return APIError(f"Model '{model_id}' not found", param="model", code="model_not_found", status=404)


def unknown_voice(voice_id: str) -> APIError:
    return APIError(f"Voice '{voice_id}' not found", param="voice", code="voice_not_found", status=404)


def unsupported_language(lang: str) -> APIError:
    return APIError(f"Unsupported language: {lang}", param="language", code="unsupported_language")


def missing_speaker_wav() -> APIError:
    return APIError(
        "Voice cloning requires speaker_wav or a registered voice",
        param="voice",
        code="missing_speaker_wav",
    )
