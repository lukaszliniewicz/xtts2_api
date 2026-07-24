import shutil
from unittest.mock import patch

from src.xtts_fastapi.coqui_compat import xtts_import_context


def test_xtts_import_context_hides_only_espeak_commands():
    def fake_which(command, *args, **kwargs):
        return f"/tools/{command}"

    with patch.object(shutil, "which", side_effect=fake_which) as original_which:
        with xtts_import_context():
            assert shutil.which("espeak-ng") is None
            assert shutil.which("espeak.exe") is None
            assert shutil.which("python") == "/tools/python"

        assert shutil.which is original_which


def test_xtts_import_context_restores_lookup_after_an_error():
    original_which = shutil.which

    try:
        with xtts_import_context():
            raise RuntimeError("import failed")
    except RuntimeError:
        pass

    assert shutil.which is original_which
