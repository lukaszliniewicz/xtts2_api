from pathlib import Path

from src.xtts_fastapi.settings import Settings


def test_settings_uses_default_models_dir_without_storage_environment(monkeypatch):
    monkeypatch.delenv("XTTS_MODELS_DIR", raising=False)
    monkeypatch.delenv("PANDRATOR_MODELS_DIR", raising=False)

    assert Settings().models_dir == Path("models")


def test_settings_uses_pandrator_models_dir_as_xtts_child(monkeypatch, tmp_path):
    monkeypatch.delenv("XTTS_MODELS_DIR", raising=False)
    storage_root = tmp_path / "pandrator-models"
    monkeypatch.setenv("PANDRATOR_MODELS_DIR", str(storage_root))

    assert Settings().models_dir == storage_root / "xtts"


def test_settings_prefers_explicit_xtts_models_dir(monkeypatch, tmp_path):
    storage_root = tmp_path / "pandrator-models"
    explicit_xtts_root = tmp_path / "dedicated-xtts-models"
    monkeypatch.setenv("PANDRATOR_MODELS_DIR", str(storage_root))
    monkeypatch.setenv("XTTS_MODELS_DIR", str(explicit_xtts_root))

    assert Settings().models_dir == explicit_xtts_root
