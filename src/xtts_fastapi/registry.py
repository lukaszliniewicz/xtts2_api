from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import unknown_model
from .settings import settings

if TYPE_CHECKING:
    from .api_models import OpenAIModel

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    HAS_WATCHDOG = True
except ImportError:
    FileSystemEventHandler = object  # type: ignore[assignment]
    Observer = None  # type: ignore[assignment]
    HAS_WATCHDOG = False

logger = logging.getLogger(__name__)

MODEL_REQUIRED_FILES = (
    "config.json",
    "model.pth",
    "speakers_xtts.pth",
    "vocab.json",
)
MODEL_DISCOVERY_IGNORE_PARTS = {".downloads", "__pycache__"}
MODEL_WATCH_RELEVANT_FILES = set(MODEL_REQUIRED_FILES)


def _normalize_model_id(model_id: str) -> str:
    return model_id.strip().replace("\\", "/").strip("/")


def _missing_required_model_files(path: Path) -> list[str]:
    return [name for name in MODEL_REQUIRED_FILES if not (path / name).is_file()]


def _is_ignored_relative_path(relative_path: Path) -> bool:
    if not relative_path.parts:
        return True

    return any(part in MODEL_DISCOVERY_IGNORE_PARTS or part.startswith(".") for part in relative_path.parts)


class _ModelWatchHandler(FileSystemEventHandler):
    def __init__(self, models_dir: Path, on_change: Callable[[str], None]):
        super().__init__()
        self._models_dir = models_dir.resolve(strict=False)
        self._on_change = on_change

    def on_any_event(self, event):
        event_type = getattr(event, "event_type", "changed")
        is_directory = bool(getattr(event, "is_directory", False))

        self._handle_path(getattr(event, "src_path", None), is_directory, event_type)
        self._handle_path(getattr(event, "dest_path", None), is_directory, f"{event_type}:dest")

    def _handle_path(self, raw_path: str | None, is_directory: bool, event_type: str) -> None:
        if not raw_path:
            return

        relative_path = self._relative_to_models_dir(Path(raw_path))
        if relative_path is None or _is_ignored_relative_path(relative_path):
            return

        if is_directory or relative_path.name in MODEL_WATCH_RELEVANT_FILES:
            self._on_change(event_type)

    def _relative_to_models_dir(self, path: Path) -> Path | None:
        candidate = path if path.is_absolute() else self._models_dir / path
        try:
            return candidate.resolve(strict=False).relative_to(self._models_dir)
        except (OSError, RuntimeError, ValueError):
            return None


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
        self._lock = threading.RLock()
        self._observer = None
        self._watch_handler: _ModelWatchHandler | None = None
        self._watch_timer: threading.Timer | None = None
        self._watch_timer_token = 0

    def _discover_from_models_dir(self, models_dir: Path) -> dict[str, ModelInfo]:
        discovered: dict[str, ModelInfo] = {}

        try:
            config_files = sorted(models_dir.rglob("config.json"), key=lambda path: str(path))
        except OSError:
            return discovered

        for config_path in config_files:
            folder = config_path.parent

            try:
                relative_folder = folder.relative_to(models_dir)
            except ValueError:
                continue

            if _is_ignored_relative_path(relative_folder):
                continue

            missing = _missing_required_model_files(folder)
            if missing:
                logger.debug(
                    "Skipping %s: missing required files: %s",
                    relative_folder.as_posix(),
                    ", ".join(missing),
                )
                continue

            model_id = _normalize_model_id(relative_folder.as_posix())
            if not model_id:
                continue
            if model_id in discovered:
                logger.warning("Duplicate model id discovered: %s (path=%s)", model_id, folder)
                continue

            config: dict = {}
            try:
                loaded = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    config = loaded
            except Exception:
                config = {}

            info = ModelInfo(model_id, folder, config)
            discovered[model_id] = info
            logger.info("Discovered model: %s", model_id)

        return discovered

    def discover(self) -> list[ModelInfo]:
        models_dir = Path(settings.models_dir)
        if not models_dir.is_dir():
            logger.warning("Models directory not found: %s", models_dir)
            with self._lock:
                self._models.clear()
            return []

        discovered = self._discover_from_models_dir(models_dir)
        with self._lock:
            self._models = discovered
            found = list(discovered.values())

        if not found:
            logger.info("No local models found in %s", models_dir)
            logger.info("Will use default Coqui model: %s", settings.default_model)

        return found

    def get(self, model_id: str) -> ModelInfo | None:
        normalized = _normalize_model_id(model_id)
        with self._lock:
            return self._models.get(normalized)

    def get_or_raise(self, model_id: str) -> ModelInfo:
        info = self.get(model_id)
        if info is None:
            raise unknown_model(model_id)
        return info

    def list_models(self) -> list[ModelInfo]:
        with self._lock:
            return list(self._models.values())

    def refresh(self) -> list[ModelInfo]:
        return self.discover()

    def start_watching(self) -> bool:
        if not settings.model_watch_enabled:
            logger.info("Model directory watcher is disabled (XTTS_MODEL_WATCH_ENABLED=false)")
            return False

        if not HAS_WATCHDOG or Observer is None:
            logger.warning("watchdog is not installed; automatic model refresh is disabled")
            return False

        models_dir = Path(settings.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if self._observer is not None:
                return True

            handler = _ModelWatchHandler(models_dir=models_dir, on_change=self._schedule_watch_refresh)
            observer = Observer()
            observer.daemon = True
            observer.schedule(handler, str(models_dir), recursive=True)
            try:
                observer.start()
            except Exception as exc:
                logger.warning("Failed to start model watcher for %s: %s", models_dir, exc)
                return False

            self._observer = observer
            self._watch_handler = handler

        logger.info("Watching models directory for changes: %s", models_dir)
        return True

    def stop_watching(self) -> None:
        with self._lock:
            timer = self._watch_timer
            self._watch_timer = None
            self._watch_timer_token += 1

            observer = self._observer
            self._observer = None
            self._watch_handler = None

        if timer is not None:
            timer.cancel()

        if observer is not None:
            try:
                observer.stop()
                observer.join(timeout=2)
            except Exception:
                logger.exception("Failed to stop model watcher cleanly")

    def _schedule_watch_refresh(self, event_type: str) -> None:
        debounce_seconds = max(float(settings.model_watch_debounce_seconds), 0.0)

        with self._lock:
            if self._watch_timer is not None:
                self._watch_timer.cancel()

            self._watch_timer_token += 1
            token = self._watch_timer_token

            timer = threading.Timer(
                debounce_seconds,
                self._run_watch_refresh,
                kwargs={"event_type": event_type, "token": token},
            )
            timer.daemon = True
            self._watch_timer = timer
            timer.start()

    def _run_watch_refresh(self, *, event_type: str, token: int) -> None:
        with self._lock:
            if token != self._watch_timer_token:
                return
            self._watch_timer = None

        try:
            refreshed = self.discover()
            logger.info(
                "Refreshed model registry from filesystem event (%s): %d model(s)",
                event_type,
                len(refreshed),
            )
        except Exception:
            logger.exception("Failed to refresh model registry from filesystem event (%s)", event_type)


registry = ModelRegistry()
