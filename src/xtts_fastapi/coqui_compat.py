"""Compatibility helpers for importing the Coqui XTTS runtime."""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


_ESPEAK_COMMANDS = frozenset({"espeak", "espeak.exe", "espeak-ng", "espeak-ng.exe"})


def _is_espeak_command(command: object) -> bool:
    return Path(str(command)).name.lower() in _ESPEAK_COMMANDS


@contextmanager
def xtts_import_context() -> Iterator[None]:
    """Keep Coqui's optional eSpeak discovery out of XTTS-only imports.

    Coqui builds a global phonemizer registry while importing its model
    modules.  That registry eagerly runs ``espeak-ng --voices`` whenever an
    executable is present, and a broken or incomplete system eSpeak install
    can therefore prevent the unrelated XTTS model from importing.  XTTS uses
    its own tokenizer, so expose eSpeak as unavailable only while Coqui's
    modules initialize.
    """

    original_which = shutil.which

    def which_without_espeak(command: str, *args: Any, **kwargs: Any) -> str | None:
        if _is_espeak_command(command):
            return None
        return original_which(command, *args, **kwargs)

    shutil.which = which_without_espeak
    try:
        yield
    finally:
        shutil.which = original_which


def import_xtts_runtime() -> tuple[Any, Any]:
    """Import and return the two Coqui classes required by the service."""

    with xtts_import_context():
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

    return XttsConfig, Xtts
