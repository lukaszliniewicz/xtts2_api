from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

APP_LOGGER_NAME = "xtts_fastapi.app"
ACCESS_LOGGER_NAME = "xtts_fastapi.access"
ERROR_LOGGER_NAME = "xtts_fastapi.errors"

DEFAULT_APP_LOG_FILE = "app.log"
DEFAULT_ACCESS_LOG_FILE = "access.log"
DEFAULT_ERROR_LOG_FILE = "errors.log"
DEFAULT_REQUEST_ID = "-"

_REQUEST_ID_CONTEXT: ContextVar[str] = ContextVar(
    "xtts_fastapi_request_id",
    default=DEFAULT_REQUEST_ID,
)

_RESERVED_RECORD_ATTRS = frozenset(logging.makeLogRecord({}).__dict__.keys()) | {
    "message",
    "asctime",
}

_LOGGING_CONFIGURED = False


def set_request_id(request_id: str) -> Token[str]:
    return _REQUEST_ID_CONTEXT.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    _REQUEST_ID_CONTEXT.reset(token)


def current_request_id() -> str:
    return _REQUEST_ID_CONTEXT.get(DEFAULT_REQUEST_ID)


def _quote_value(value: object) -> str:
    text = str(value)
    text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{text}"'


class KeyValueFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return timestamp.isoformat(timespec="milliseconds")

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        fields: dict[str, object] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", DEFAULT_REQUEST_ID),
            "msg": message,
        }

        extras: dict[str, object] = {}
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_ATTRS or key == "request_id":
                continue
            if key.startswith("_"):
                continue
            if value is None:
                continue
            extras[key] = value

        for key in sorted(extras.keys()):
            fields[key] = extras[key]

        if record.exc_info:
            fields["exc"] = self.formatException(record.exc_info)

        return " ".join(f"{key}={_quote_value(value)}" for key, value in fields.items())


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", None):
            record.request_id = current_request_id()
        return True


def _coerce_level(level_name: str) -> int:
    candidate = getattr(logging, level_name.upper(), None)
    if isinstance(candidate, int):
        return candidate
    return logging.INFO


def _logger_has_marker(logger: logging.Logger, marker: str) -> bool:
    for handler in logger.handlers:
        if getattr(handler, "_xtts_marker", None) == marker:
            return True
    return False


def _make_rotating_handler(
    *,
    log_path: Path,
    level: int,
    marker: str,
    max_bytes: int,
    backup_count: int,
    formatter: logging.Formatter,
    request_id_filter: logging.Filter,
) -> logging.Handler | None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_path,
            maxBytes=max(1, int(max_bytes)),
            backupCount=max(1, int(backup_count)),
            encoding="utf-8",
        )
    except OSError:
        logging.getLogger(__name__).exception(
            "Failed to create log file handler",
            extra={"path": str(log_path)},
        )
        return None

    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler.addFilter(request_id_filter)
    handler._xtts_marker = marker
    return handler


def configure_file_logging(
    *,
    logs_dir: Path,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    app_log_file: str = DEFAULT_APP_LOG_FILE,
    access_log_file: str = DEFAULT_ACCESS_LOG_FILE,
    error_log_file: str = DEFAULT_ERROR_LOG_FILE,
) -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    base_level = _coerce_level(level)
    formatter = KeyValueFormatter()
    request_id_filter = RequestIdFilter()
    configured_any = False

    app_handler = _make_rotating_handler(
        log_path=Path(logs_dir) / app_log_file,
        level=base_level,
        marker="xtts_app_file",
        max_bytes=max_bytes,
        backup_count=backup_count,
        formatter=formatter,
        request_id_filter=request_id_filter,
    )
    access_handler = _make_rotating_handler(
        log_path=Path(logs_dir) / access_log_file,
        level=logging.INFO,
        marker="xtts_access_file",
        max_bytes=max_bytes,
        backup_count=backup_count,
        formatter=formatter,
        request_id_filter=request_id_filter,
    )
    error_handler = _make_rotating_handler(
        log_path=Path(logs_dir) / error_log_file,
        level=logging.ERROR,
        marker="xtts_error_file",
        max_bytes=max_bytes,
        backup_count=backup_count,
        formatter=formatter,
        request_id_filter=request_id_filter,
    )

    if app_handler is not None:
        app_logger = logging.getLogger("src.xtts_fastapi")
        app_logger.setLevel(base_level)
        if not _logger_has_marker(app_logger, "xtts_app_file"):
            app_logger.addHandler(app_handler)
            configured_any = True

        package_logger = logging.getLogger("xtts_fastapi")
        package_logger.setLevel(base_level)
        if not _logger_has_marker(package_logger, "xtts_app_file"):
            package_logger.addHandler(app_handler)
            configured_any = True

        service_app_logger = logging.getLogger(APP_LOGGER_NAME)
        service_app_logger.setLevel(base_level)
        service_app_logger.propagate = False
        if not _logger_has_marker(service_app_logger, "xtts_app_file"):
            service_app_logger.addHandler(app_handler)
            configured_any = True

        uvicorn_error_logger = logging.getLogger("uvicorn.error")
        uvicorn_error_logger.setLevel(base_level)
        if not _logger_has_marker(uvicorn_error_logger, "xtts_app_file"):
            uvicorn_error_logger.addHandler(app_handler)
            configured_any = True

    if access_handler is not None:
        access_logger = logging.getLogger(ACCESS_LOGGER_NAME)
        access_logger.setLevel(logging.INFO)
        access_logger.propagate = False
        if not _logger_has_marker(access_logger, "xtts_access_file"):
            access_logger.addHandler(access_handler)
            configured_any = True

        uvicorn_access_logger = logging.getLogger("uvicorn.access")
        uvicorn_access_logger.setLevel(logging.INFO)
        if not _logger_has_marker(uvicorn_access_logger, "xtts_access_file"):
            uvicorn_access_logger.addHandler(access_handler)
            configured_any = True

    if error_handler is not None:
        error_logger = logging.getLogger(ERROR_LOGGER_NAME)
        error_logger.setLevel(logging.ERROR)
        error_logger.propagate = False
        if not _logger_has_marker(error_logger, "xtts_error_file"):
            error_logger.addHandler(error_handler)
            configured_any = True

        uvicorn_error_logger = logging.getLogger("uvicorn.error")
        if not _logger_has_marker(uvicorn_error_logger, "xtts_error_file"):
            uvicorn_error_logger.addHandler(error_handler)
            configured_any = True

    _LOGGING_CONFIGURED = configured_any
