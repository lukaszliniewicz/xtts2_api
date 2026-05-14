from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import unknown_model
from .settings import settings

if TYPE_CHECKING:
    from .api_models import OpenAIModel

logger = logging.getLogger(__name__)


class ModelInfo:
    def __init__(self, model_id: str, path: Path, config: dict | None = None):
        self.model_id = model_id
        self.path = path
        self.config = config or {}
        self.is_xtts = self._detect_xtts()

    def _detect_xtts(self) -> bool:
        if self.config.get("model") == "xtts":
            return True
        return "xtts" in self.model_id.lower()

    def to_openai(self) -> OpenAIModel:
        from .api_models import OpenAIModel

        return OpenAIModel(id=self.model_id, owned_by="xtts-fapi")


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, ModelInfo] = {}

    def discover(self) -> list[ModelInfo]:
        self._models.clear()
        models_dir = Path(settings.models_dir)
        if not models_dir.is_dir():
            logger.warning("Models directory not found: %s", models_dir)
            return []

        found = []
        for folder in models_dir.iterdir():
            if not folder.is_dir():
                continue
            model_id = folder.name
            config_path = folder / "config.json"
            if not config_path.is_file():
                logger.debug("Skipping %s: no config.json", model_id)
                continue
            try:
                config = json.loads(config_path.read_text())
            except Exception:
                config = {}
            info = ModelInfo(model_id, folder, config)
            self._models[model_id] = info
            found.append(info)
            logger.info("Discovered model: %s", model_id)

        if not found:
            logger.info("No local models found in %s", models_dir)
            logger.info("Will use default Coqui model: %s", settings.default_model)

        return found

    def get(self, model_id: str) -> ModelInfo | None:
        return self._models.get(model_id)

    def get_or_raise(self, model_id: str) -> ModelInfo:
        info = self.get(model_id)
        if info is None:
            raise unknown_model(model_id)
        return info

    def list_models(self) -> list[ModelInfo]:
        return list(self._models.values())

    def refresh(self) -> list[ModelInfo]:
        return self.discover()


registry = ModelRegistry()
