from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.xtts_fastapi.engine import engine
from src.xtts_fastapi.main import app
from src.xtts_fastapi.registry import registry
from src.xtts_fastapi.settings import settings

client = TestClient(app, client=("127.0.0.1", 50001))


def _write_bundle(path, *, complete: bool = True):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model":"xtts"}', encoding="utf-8")
    (path / "model.pth").write_bytes(b"weights")
    if complete:
        (path / "speakers_xtts.pth").write_bytes(b"speakers")
        (path / "vocab.json").write_text('{"en":0}', encoding="utf-8")


@pytest.fixture
def models_root(tmp_path, monkeypatch):
    original_models_dir = settings.models_dir
    original_models = engine._models.copy()
    root = tmp_path / "models"
    monkeypatch.setattr(settings, "models_dir", root)
    with engine._registry_lock:
        engine._models.clear()
    engine.clear_conditioning_cache()
    registry.refresh()
    try:
        yield root
    finally:
        monkeypatch.setattr(settings, "models_dir", original_models_dir)
        with engine._registry_lock:
            engine._models.clear()
            engine._models.update(original_models)
        engine.clear_conditioning_cache()
        registry.refresh()


def test_model_list_marks_builtin_and_removable_local_metadata(models_root):
    _write_bundle(models_root / "custom" / "acme-voice")
    registry.refresh()

    listed = {item["id"]: item for item in client.get("/v1/models").json()["data"]}

    assert listed[settings.default_model]["is_default"] is True
    assert listed[settings.default_model]["is_local"] is False
    assert listed[settings.default_model]["removable"] is False
    assert listed[settings.default_model]["source"] == "builtin"
    assert listed[settings.default_model]["relative_path"] is None

    local = listed["custom/acme-voice"]
    assert local["is_default"] is False
    assert local["is_local"] is True
    assert local["removable"] is True
    assert local["source"] == "local"
    assert local["relative_path"] == "custom/acme-voice"
    assert local["bundle_complete"] is True


def test_delete_nested_model_evicts_only_that_wrapper_and_refreshes_registry(models_root):
    _write_bundle(models_root / "custom" / "acme-voice")
    _write_bundle(models_root / "other-voice")
    registry.refresh()

    class FakeWrapper:
        def __init__(self, model):
            self.xtts_model = model
            self.unloaded = False

        def unload(self):
            self.unloaded = True
            self.xtts_model = None

    target_model = object()
    other_model = object()
    target_wrapper = FakeWrapper(target_model)
    other_wrapper = FakeWrapper(other_model)
    with engine._registry_lock:
        engine._models.update(
            {
                "custom/acme-voice": target_wrapper,
                "other-voice": other_wrapper,
            }
        )
    with engine._latent_cache_lock:
        engine._latent_cache[(id(target_model), "target")] = ("target",)
        engine._latent_cache[(id(other_model), "other")] = ("other",)

    response = client.delete("/v1/models/custom/acme-voice")

    assert response.status_code == 200
    assert response.json() == {
        "id": "custom/acme-voice",
        "object": "model",
        "deleted": True,
        "evicted": True,
    }
    assert not (models_root / "custom" / "acme-voice").exists()
    assert registry.get("custom/acme-voice") is None
    assert target_wrapper.unloaded is True
    assert other_wrapper.unloaded is False
    with engine._latent_cache_lock:
        assert (id(target_model), "target") not in engine._latent_cache
        assert engine._latent_cache[(id(other_model), "other")] == ("other",)


@pytest.mark.parametrize(
    ("model_id", "status", "code"),
    [
        ("missing", 404, "model_not_found"),
        ("../outside", 400, "invalid_model_id"),
        ("", 400, "invalid_model_id"),
    ],
)
def test_delete_rejects_missing_traversal_and_root(models_root, model_id, status, code):
    if model_id == "../outside":
        response = client.delete("/v1/models/%2E%2E/outside")
        assert response.status_code == status
        assert response.json()["error"]["code"] == code
        return

    response = client.delete("/v1/models" if not model_id else f"/v1/models/{model_id}")

    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_delete_rejects_default_and_incomplete_bundles(models_root):
    incomplete = models_root / "incomplete"
    _write_bundle(incomplete, complete=False)
    registry.refresh()

    default_response = client.delete(f"/v1/models/{settings.default_model}")
    alias_response = client.delete("/v1/models/xtts_v2")
    incomplete_response = client.delete("/v1/models/incomplete")

    assert default_response.status_code == 403
    assert default_response.json()["error"]["code"] == "default_model_protected"
    assert alias_response.status_code == 403
    assert alias_response.json()["error"]["code"] == "default_model_protected"
    assert incomplete_response.status_code == 409
    assert incomplete_response.json()["error"]["code"] == "model_bundle_incomplete"
    assert incomplete.exists()


def test_delete_rejects_symlink_escape_without_touching_external_files(models_root, tmp_path):
    models_root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    _write_bundle(outside)
    (models_root / "linked").symlink_to(outside, target_is_directory=True)

    response = client.delete("/v1/models/linked")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_model_id"
    assert (outside / "model.pth").is_file()


def test_delete_during_held_model_lock_waits_then_removes(models_root):
    _write_bundle(models_root / "locked")
    registry.refresh()

    entered = engine._get_lock("locked").acquire(blocking=False)
    assert entered is True
    try:
        # The lock is exact and re-entrant only by thread, so a background
        # request must wait.  A short timeout keeps this test deterministic.
        import threading

        result = []

        def run_delete():
            result.append(client.delete("/v1/models/locked"))

        worker = threading.Thread(target=run_delete)
        worker.start()
        worker.join(timeout=0.05)
        assert worker.is_alive()
    finally:
        engine._get_lock("locked").release()

    worker.join(timeout=2)
    assert not worker.is_alive()
    assert result[0].status_code == 200
    assert not (models_root / "locked").exists()
