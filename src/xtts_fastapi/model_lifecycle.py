from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from uuid import uuid4

from .api_models import ModelDeletedResponse
from .engine import engine
from .errors import APIError
from .model_uploads import (
    _remove_directory,
    _validate_model_id,
)
from .registry import MODEL_REQUIRED_FILES, registry
from .settings import settings


def _missing_bundle_files(path: Path) -> list[str]:
    return [name for name in MODEL_REQUIRED_FILES if not (path / name).is_file()]


def _contains_symlink(path: Path) -> bool:
    try:
        return any(candidate.is_symlink() for candidate in path.rglob("*"))
    except (OSError, RuntimeError) as exc:
        raise APIError(
            "Could not inspect the model bundle safely",
            param="model_id",
            code="model_bundle_unavailable",
            status=500,
        ) from exc


def _cleanup_stale_deletion_staging(downloads_dir: Path, root: Path) -> None:
    """Remove only old hidden deletion staging directories under the root."""
    cutoff = time.time() - 24 * 60 * 60
    try:
        candidates = list(downloads_dir.iterdir())
    except OSError:
        return

    for candidate in candidates:
        if not candidate.name.startswith(".deleting-") or candidate.is_symlink():
            continue
        try:
            if candidate.stat().st_mtime >= cutoff:
                continue
        except OSError:
            continue
        _remove_directory(candidate, root)


def _is_protected_default(model_id: str, target: Path) -> bool:
    if model_id in {
        settings.default_model,
        "xtts_v2",
        "tts_models/multilingual/multi-dataset/xtts_v2",
    }:
        return True

    try:
        default_path = (Path(settings.models_dir) / settings.default_model_local_dir).resolve(strict=False)
        return target.resolve(strict=False) == default_path
    except (OSError, RuntimeError):
        return True


def _validate_deletable_bundle(model_id: str, root: Path, target: Path) -> None:
    if _is_protected_default(model_id, target):
        raise APIError(
            f"The default model '{model_id}' is protected and cannot be deleted",
            param="model_id",
            code="default_model_protected",
            status=403,
        )

    if not target.exists():
        raise APIError(
            f"Model '{model_id}' not found",
            param="model_id",
            code="model_not_found",
            status=404,
        )
    if not target.is_dir():
        raise APIError(
            f"Model '{model_id}' is not a model bundle directory",
            param="model_id",
            code="model_bundle_incomplete",
            status=409,
        )

    try:
        target.resolve(strict=True).relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise APIError(
            "Model bundle must remain inside the configured models directory",
            param="model_id",
            code="model_bundle_symlink",
            status=400,
        ) from exc

    if _contains_symlink(target):
        raise APIError(
            "Model bundles containing symlinks cannot be deleted",
            param="model_id",
            code="model_bundle_symlink",
            status=400,
        )

    missing = _missing_bundle_files(target)
    if missing:
        raise APIError(
            f"Model '{model_id}' is incomplete; missing required file(s): {', '.join(missing)}",
            param="model_id",
            code="model_bundle_incomplete",
            status=409,
        )


def delete_model(model_id: str) -> ModelDeletedResponse:
    """Atomically hide and remove one complete local model bundle.

    The exact model lock is held across wrapper eviction and the filesystem
    rename/removal.  Inference uses the same lock and revalidates its registry
    entry after acquiring it, so a request queued before deletion cannot load a
    stale path after deletion completes.
    """
    normalized_id, root, downloads_dir, target = _validate_model_id(model_id)
    _validate_deletable_bundle(normalized_id, root, target)

    with engine.model_lock(normalized_id):
        _validate_deletable_bundle(normalized_id, root, target)
        _cleanup_stale_deletion_staging(downloads_dir, root)

        evicted = engine._evict_model_locked(normalized_id)
        trash = downloads_dir / f".deleting-{uuid4().hex}"
        try:
            os.replace(target, trash)
            shutil.rmtree(trash)
        except OSError as exc:
            # A renamed-but-not-removed bundle stays hidden in .downloads and
            # can be cleaned by a later request.  It is never republished as a
            # partial model.
            registry.refresh()
            raise APIError(
                f"Model '{normalized_id}' could not be deleted safely",
                param="model_id",
                code="model_delete_failed",
                status=500,
            ) from exc

        registry.refresh()
        return ModelDeletedResponse(id=normalized_id, evicted=evicted)
