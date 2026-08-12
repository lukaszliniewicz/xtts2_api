from __future__ import annotations

import json
import os
import shutil
import unicodedata
from pathlib import Path, PureWindowsPath
from uuid import uuid4

from fastapi import UploadFile

from .api_models import ModelUploadResponse
from .errors import APIError
from .registry import MODEL_DISCOVERY_IGNORE_PARTS, MODEL_REQUIRED_FILES, registry
from .settings import settings

WINDOWS_RESERVED_PARTS = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
INVALID_MODEL_ID_CHARS = set('<>:"|?*\\')


def _invalid_model_id(message: str) -> APIError:
    return APIError(message, param="model_id", code="invalid_model_id")


def _models_root() -> tuple[Path, Path]:
    configured_root = Path(settings.models_dir)
    try:
        configured_root.mkdir(parents=True, exist_ok=True)
        root = configured_root.resolve(strict=True)
    except OSError as exc:
        raise APIError(
            f"Could not prepare models directory: {exc}",
            param="model_id",
            code="model_storage_unavailable",
            status=500,
        ) from exc

    if not root.is_dir():
        raise APIError(
            "Configured models directory is not a directory",
            param="model_id",
            code="model_storage_unavailable",
            status=500,
        )

    downloads_dir = root / ".downloads"
    if downloads_dir.is_symlink():
        raise APIError(
            "Model staging directory must not be a symlink",
            param="model_id",
            code="model_storage_unavailable",
            status=500,
        )
    try:
        downloads_dir.mkdir(exist_ok=True)
    except OSError as exc:
        raise APIError(
            f"Could not prepare model staging directory: {exc}",
            param="model_id",
            code="model_storage_unavailable",
            status=500,
        ) from exc
    return root, downloads_dir


def _validate_model_id(model_id: str) -> tuple[str, Path, Path, Path]:
    candidate_id = str(model_id or "")
    if not candidate_id or candidate_id != candidate_id.strip():
        raise _invalid_model_id("model_id must be a non-empty relative identifier without surrounding spaces")
    if any(unicodedata.category(character).startswith("C") for character in candidate_id):
        raise _invalid_model_id("model_id must not contain control characters")
    if "\\" in candidate_id or candidate_id.startswith("/") or PureWindowsPath(candidate_id).is_absolute():
        raise _invalid_model_id("model_id must be a relative slash-separated path")

    parts = candidate_id.split("/")
    if any(not part for part in parts):
        raise _invalid_model_id("model_id must not contain empty path parts")

    for part in parts:
        windows_device_basename = part.split(".", maxsplit=1)[0].casefold()
        if (
            part in {".", ".."}
            or part.startswith(".")
            or part != part.rstrip(". ")
            or part in MODEL_DISCOVERY_IGNORE_PARTS
            or windows_device_basename in WINDOWS_RESERVED_PARTS
            or any(character in INVALID_MODEL_ID_CHARS for character in part)
        ):
            raise _invalid_model_id("model_id contains a reserved, hidden, or unsafe path part")

    root, downloads_dir = _models_root()
    target = root.joinpath(*parts)
    try:
        target.resolve(strict=False).relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise _invalid_model_id("model_id must resolve inside the configured models directory") from exc

    current = root
    for index, part in enumerate(parts):
        current = current / part
        if current.is_symlink():
            raise _invalid_model_id("model_id must not pass through a symlink")
        if index < len(parts) - 1 and current.exists() and not current.is_dir():
            raise _invalid_model_id("model_id has a non-directory parent path")

    return candidate_id, root, downloads_dir, target


def _validate_upload_names(files: list[UploadFile]) -> dict[str, UploadFile]:
    uploaded_by_name: dict[str, UploadFile] = {}
    for upload in files:
        filename = upload.filename or ""
        if filename not in MODEL_REQUIRED_FILES:
            raise APIError(
                f"Unexpected model bundle file: {filename or '<unnamed>'}",
                param="files",
                code="unexpected_model_file",
            )
        if filename in uploaded_by_name:
            raise APIError(
                f"Duplicate model bundle file: {filename}",
                param="files",
                code="duplicate_model_file",
            )
        uploaded_by_name[filename] = upload

    missing = [name for name in MODEL_REQUIRED_FILES if name not in uploaded_by_name]
    if missing:
        raise APIError(
            f"Missing required model bundle file(s): {', '.join(missing)}",
            param="files",
            code="missing_model_files",
        )
    return uploaded_by_name


async def _copy_upload(
    upload: UploadFile,
    destination: Path,
    *,
    filename: str,
    total_bytes: int,
    file_limit: int,
) -> tuple[int, int]:
    written = 0
    chunk_size = settings.model_upload_chunk_size
    with destination.open("xb") as destination_file:
        while chunk := await upload.read(chunk_size):
            next_file_size = written + len(chunk)
            next_total_size = total_bytes + len(chunk)
            if next_file_size > file_limit:
                raise APIError(
                    f"{filename} exceeds its upload size limit of {file_limit} bytes",
                    param="files",
                    code="model_upload_too_large",
                    status=413,
                )
            if next_total_size > settings.model_upload_max_total_bytes:
                raise APIError(
                    f"Model bundle exceeds its total upload size limit of "
                    f"{settings.model_upload_max_total_bytes} bytes",
                    param="files",
                    code="model_upload_too_large",
                    status=413,
                )
            destination_file.write(chunk)
            written = next_file_size
            total_bytes = next_total_size

    if written == 0:
        raise APIError(
            f"Required model bundle file is empty: {filename}",
            param="files",
            code="empty_model_file",
        )
    return written, total_bytes


def _validate_xtts_config(config_path: Path) -> None:
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise APIError(
            "config.json must contain valid UTF-8 JSON",
            param="files",
            code="invalid_model_config",
            status=422,
        ) from exc

    if not isinstance(config, dict):
        raise APIError(
            "config.json must contain a JSON object",
            param="files",
            code="invalid_model_config",
            status=422,
        )

    model_type = config.get("model")
    if not isinstance(model_type, str) or model_type.casefold() != "xtts":
        raise APIError(
            "config.json must identify an XTTS model (\"model\": \"xtts\")",
            param="files",
            code="invalid_model_config",
            status=422,
        )


def _remove_directory(path: Path, root: Path) -> None:
    """Best-effort cleanup that will never follow a path substituted with a symlink."""
    try:
        if path.is_symlink() or not path.is_dir():
            return
        path.resolve(strict=True).relative_to(root)
        shutil.rmtree(path)
    except (OSError, RuntimeError, ValueError):
        return


async def install_model_upload(model_id: str, files: list[UploadFile]) -> ModelUploadResponse:
    """Stream an XTTS bundle to hidden staging, then atomically publish it."""
    staging_dir: Path | None = None
    target: Path | None = None
    root: Path | None = None
    promoted = False

    try:
        normalized_id, root, downloads_dir, target = _validate_model_id(model_id)
        uploaded_by_name = _validate_upload_names(files)

        if target.exists() or target.is_symlink():
            raise APIError(
                f"Model '{normalized_id}' already exists; overwriting models is not supported",
                param="model_id",
                code="model_already_exists",
                status=409,
            )

        staging_dir = downloads_dir / uuid4().hex
        staging_dir.mkdir()
        total_bytes = 0

        for filename in MODEL_REQUIRED_FILES:
            file_limit = settings.model_upload_max_file_bytes
            if filename == "config.json":
                file_limit = min(file_limit, settings.model_upload_max_config_bytes)
            _, total_bytes = await _copy_upload(
                uploaded_by_name[filename],
                staging_dir / filename,
                filename=filename,
                total_bytes=total_bytes,
                file_limit=file_limit,
            )
            if filename == "config.json":
                _validate_xtts_config(staging_dir / filename)

        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise APIError(
                f"Model '{normalized_id}' already exists; overwriting models is not supported",
                param="model_id",
                code="model_already_exists",
                status=409,
            )
        os.replace(staging_dir, target)
        promoted = True

        registry.refresh()
        if registry.get(normalized_id) is None:
            raise RuntimeError("published model was not discovered during synchronous registry refresh")

        return ModelUploadResponse(id=normalized_id, bytes=total_bytes)
    except APIError:
        if promoted and target is not None and root is not None:
            _remove_directory(target, root)
            registry.refresh()
        raise
    except Exception as exc:
        if promoted and target is not None and root is not None:
            _remove_directory(target, root)
            registry.refresh()
        raise APIError(
            "Model upload failed before it could be installed",
            param="files",
            code="model_upload_failed",
            status=500,
        ) from exc
    finally:
        if staging_dir is not None and root is not None:
            _remove_directory(staging_dir, root)
        for upload in files:
            try:
                await upload.close()
            except Exception:
                continue
