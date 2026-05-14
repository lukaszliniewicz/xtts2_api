from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .settings import settings

if TYPE_CHECKING:
    from .api_models import Voice, VoiceCreateResponse, VoiceFile

logger = logging.getLogger(__name__)


class VoiceStore:
    def __init__(self):
        self._base_dir = Path(settings.voices_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _voice_path(self, voice_id: str) -> Path:
        return self._base_dir / voice_id

    def _meta_path(self, voice_id: str) -> Path:
        return self._voice_path(voice_id) / "meta.json"

    def create(self, voice_id: str, files: list[tuple[str, bytes]], model: str | None = None, language: str | None = None) -> VoiceCreateResponse:
        vpath = self._voice_path(voice_id)
        if vpath.exists():
            shutil.rmtree(vpath)
        vpath.mkdir(parents=True)

        sample_count = 0
        file_list: list[VoiceFile] = []
        for name, data in files:
            stem = Path(name).stem
            dest = vpath / f"{stem}.wav"
            dest.write_bytes(data)
            file_list.append({"filename": dest.name, "size": len(data)})
            sample_count += 1

        created = int(time.time())
        meta = {
            "voice_id": voice_id,
            "created": created,
            "model": model,
            "language": language,
            "files": file_list,
        }
        self._meta_path(voice_id).write_text(json.dumps(meta, indent=2))

        from .api_models import VoiceCreateResponse

        return VoiceCreateResponse(
            id=voice_id,
            model=model,
            language=language,
            sample_count=sample_count,
            created=created,
        )

    def get(self, voice_id: str) -> dict | None:
        mpath = self._meta_path(voice_id)
        if not mpath.is_file():
            return None
        return json.loads(mpath.read_text())

    def delete(self, voice_id: str) -> bool:
        vpath = self._voice_path(voice_id)
        if not vpath.exists():
            return False
        shutil.rmtree(vpath)
        return True

    def list_all(self) -> list[Voice]:
        from .api_models import Voice

        voices: list[Voice] = []
        if not self._base_dir.is_dir():
            return voices
        for entry in self._base_dir.iterdir():
            if not entry.is_dir():
                continue
            meta_path = entry / "meta.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text())
            voices.append(Voice(**meta))
        return voices

    def has_voice(self, voice_id: str) -> bool:
        return self._voice_path(voice_id).is_dir()

    def get_sample_paths(self, voice_id: str) -> list[Path]:
        vpath = self._voice_path(voice_id)
        if not vpath.is_dir():
            return []
        return sorted(vpath.glob("*.wav"))


voice_store = VoiceStore()
