from fastapi.testclient import TestClient

from src.xtts_fastapi.main import app
from src.xtts_fastapi.model_loader import _coqui_cache_folder_names, _resolve_default_hf_source

client = TestClient(app)


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
