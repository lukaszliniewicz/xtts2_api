from fastapi.testclient import TestClient

from src.xtts_fastapi.main import app
from src.xtts_fastapi.model_loader import (
    _coqui_cache_folder_names,
    _find_default_model_in_tree,
    _has_complete_default_model,
    _resolve_default_hf_source,
)
from src.xtts_fastapi.registry import ModelRegistry

client = TestClient(app)


def _write_complete_model_bundle(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.pth").write_bytes(b"weights")
    (path / "speakers_xtts.pth").write_bytes(b"speakers")
    (path / "vocab.json").write_text("{}", encoding="utf-8")


def test_list_models_empty():
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)


def test_refresh_models():
    resp = client.post("/admin/models/refresh")
    assert resp.status_code == 200
    data = resp.json()
    assert "refreshed" in data


def test_coqui_cache_folder_names_for_full_model_id():
    names = _coqui_cache_folder_names("tts_models/multilingual/multi-dataset/xtts_v2")
    assert "tts_models--multilingual--multi-dataset--xtts_v2" in names


def test_coqui_cache_folder_names_for_short_xtts_id():
    names = _coqui_cache_folder_names("xtts_v2")
    assert "tts_models--multilingual--multi-dataset--xtts_v2" in names


def test_resolve_default_hf_source_for_xtts_default():
    source = _resolve_default_hf_source("tts_models/multilingual/multi-dataset/xtts_v2")
    assert source == ("coqui/XTTS-v2", "main")


def test_resolve_default_hf_source_parses_version():
    source = _resolve_default_hf_source("xtts_v2.0.2")
    assert source == ("coqui/XTTS-v2", "v2.0.2")


def test_has_complete_default_model_requires_weights(tmp_path):
    model_dir = tmp_path / "XTTS_2.0.2"
    model_dir.mkdir(parents=True)

    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "vocab.json").write_text("{}", encoding="utf-8")
    (model_dir / "speakers_xtts.pth").write_bytes(b"speakers")

    assert _has_complete_default_model(model_dir) is False

    (model_dir / "model.pth").write_bytes(b"weights")
    assert _has_complete_default_model(model_dir) is True


def test_find_default_model_in_tree_finds_nested_complete_bundle(tmp_path):
    models_root = tmp_path / "models"
    incomplete = models_root / "partial_xtts"
    incomplete.mkdir(parents=True)
    (incomplete / "config.json").write_text("{}", encoding="utf-8")

    complete = models_root / "nested" / "xtts_v2"
    complete.mkdir(parents=True)
    (complete / "config.json").write_text("{}", encoding="utf-8")
    (complete / "model.pth").write_bytes(b"weights")
    (complete / "speakers_xtts.pth").write_bytes(b"speakers")
    (complete / "vocab.json").write_text("{}", encoding="utf-8")

    found = _find_default_model_in_tree(models_root, "tts_models/multilingual/multi-dataset/xtts_v2")
    assert found == complete


def test_registry_discover_supports_nested_model_folders(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    _write_complete_model_bundle(models_root / "flat_model")
    _write_complete_model_bundle(models_root / "nested" / "sub" / "model_x")

    monkeypatch.setattr("src.xtts_fastapi.registry.settings.models_dir", models_root)

    local_registry = ModelRegistry()
    model_ids = sorted(model.model_id for model in local_registry.discover())

    assert model_ids == ["flat_model", "nested/sub/model_x"]


def test_registry_discover_requires_all_model_files(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    _write_complete_model_bundle(models_root / "complete")

    incomplete = models_root / "incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    (incomplete / "config.json").write_text("{}", encoding="utf-8")
    (incomplete / "model.pth").write_bytes(b"weights")

    monkeypatch.setattr("src.xtts_fastapi.registry.settings.models_dir", models_root)

    local_registry = ModelRegistry()
    model_ids = [model.model_id for model in local_registry.discover()]

    assert model_ids == ["complete"]
