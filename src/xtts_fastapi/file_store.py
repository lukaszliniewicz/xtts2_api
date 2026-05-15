from __future__ import annotations

import json
import secrets
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import APIError
from .settings import settings

if TYPE_CHECKING:
    from .api_models import FileDeletedResponse, FileListResponse, FileObject


class FileStore:
    def __init__(self):
        self._base_dir = Path(settings.files_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, file_id: str) -> Path:
        return self._base_dir / file_id

    def _meta_path(self, file_id: str) -> Path:
        return self._file_path(file_id) / "meta.json"

    def _content_path(self, file_id: str) -> Path:
        return self._file_path(file_id) / "payload.bin"

    def _new_file_id(self) -> str:
        while True:
            file_id = f"file-{secrets.token_hex(12)}"
            if not self._file_path(file_id).exists():
                return file_id

    def _load_meta(self, file_id: str) -> dict | None:
        meta_path = self._meta_path(file_id)
        if not meta_path.is_file():
            return None
        return json.loads(meta_path.read_text())

    def create(self, filename: str, data: bytes, purpose: str) -> FileObject:
        if not data:
            raise APIError("Uploaded file is empty", param="file", code="empty_file")
        if not purpose:
            raise APIError("purpose is required", param="purpose", code="missing_purpose")

        from .api_models import FileObject

        safe_filename = Path(filename).name if filename else "upload.bin"
        if not safe_filename:
            safe_filename = "upload.bin"

        file_id = self._new_file_id()
        created_at = int(time.time())

        file_dir = self._file_path(file_id)
        file_dir.mkdir(parents=True, exist_ok=False)
        self._content_path(file_id).write_bytes(data)

        file_obj = FileObject(
            id=file_id,
            bytes=len(data),
            created_at=created_at,
            filename=safe_filename,
            purpose=purpose,
            status="processed",
            expires_at=None,
            status_details=None,
        )
        self._meta_path(file_id).write_text(json.dumps(file_obj.model_dump(), indent=2))
        return file_obj

    def get(self, file_id: str) -> FileObject | None:
        from .api_models import FileObject

        meta = self._load_meta(file_id)
        if meta is None:
            return None
        return FileObject(**meta)

    def get_content_path(self, file_id: str) -> Path | None:
        if self.get(file_id) is None:
            return None
        content_path = self._content_path(file_id)
        if not content_path.is_file():
            return None
        return content_path

    def get_content(self, file_id: str) -> bytes | None:
        content_path = self.get_content_path(file_id)
        if content_path is None:
            return None
        return content_path.read_bytes()

    def list_all(
        self,
        *,
        limit: int = 100,
        after: str | None = None,
        order: str = "desc",
        purpose: str | None = None,
    ) -> FileListResponse:
        from .api_models import FileListResponse

        files = []
        if self._base_dir.is_dir():
            for entry in self._base_dir.iterdir():
                if not entry.is_dir():
                    continue
                file_obj = self.get(entry.name)
                if file_obj is None:
                    continue
                if purpose is not None and file_obj.purpose != purpose:
                    continue
                files.append(file_obj)

        reverse = order != "asc"
        files.sort(key=lambda item: (item.created_at, item.id), reverse=reverse)

        if after is not None:
            for idx, file_obj in enumerate(files):
                if file_obj.id == after:
                    files = files[idx + 1 :]
                    break

        sliced = files[:limit]
        has_more = len(files) > limit
        return FileListResponse(object="list", data=sliced, has_more=has_more)

    def delete(self, file_id: str) -> bool:
        file_path = self._file_path(file_id)
        if not file_path.exists():
            return False
        shutil.rmtree(file_path)
        return True

    def delete_response(self, file_id: str) -> FileDeletedResponse:
        from .api_models import FileDeletedResponse

        return FileDeletedResponse(id=file_id, object="file", deleted=True)


file_store = FileStore()
