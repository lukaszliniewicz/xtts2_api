import json

from fastapi.testclient import TestClient

from src.xtts_fastapi.main import app
from src.xtts_fastapi.settings import settings
from src.xtts_fastapi.voices import VoiceStore

client = TestClient(app)


def test_list_voices_empty():
    resp = client.get("/v1/voices")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"


def test_create_voice_no_files():
    resp = client.post("/v1/voices")
    assert resp.status_code in (400, 422)


def test_create_and_delete_voice():
    wav_data = b"\x00" * 1024
    resp = client.post(
        "/v1/voices",
        files={"files": ("test_sample.wav", wav_data, "audio/wav")},
        data={"voice_id": "test_voice"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "test_voice"
    assert data["sample_count"] >= 1

    resp = client.get("/v1/voices")
    assert resp.status_code == 200
    ids = [v["voice_id"] for v in resp.json()["data"]]
    assert "test_voice" in ids

    resp = client.delete("/v1/voices/test_voice")
    assert resp.status_code == 200

    resp = client.delete("/v1/voices/test_voice")
    assert resp.status_code == 404


def test_create_voice_auto_id():
    wav_data = b"\x00" * 1024
    resp = client.post(
        "/v1/voices",
        files={"files": ("my_custom_name.wav", wav_data, "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "my_custom_name"
    client.delete(f"/v1/voices/{data['id']}")


def test_create_voice_auto_id_normalizes_filename():
    wav_data = b"\x00" * 1024
    resp = client.post(
        "/v1/voices",
        files={"files": ("My Cool Voice (v1).wav", wav_data, "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "my-cool-voice-v1"
    client.delete(f"/v1/voices/{data['id']}")


def test_create_voice_custom_id_is_normalized():
    wav_data = b"\x00" * 1024
    resp = client.post(
        "/v1/voices",
        files={"files": ("sample.wav", wav_data, "audio/wav")},
        data={"voice_id": "  Team Voice #1  "},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "team-voice-1"
    client.delete(f"/v1/voices/{data['id']}")


def test_register_staged_voices_creates_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "voices_dir", tmp_path)
    voice_dir = tmp_path / "staged_voice"
    voice_dir.mkdir(parents=True, exist_ok=True)
    (voice_dir / "sample_b.wav").write_bytes(b"\x00" * 256)
    (voice_dir / "sample_a.WAV").write_bytes(b"\x00" * 128)

    store = VoiceStore()
    registered = store.register_staged_voices()

    assert registered == 1
    meta_path = voice_dir / "meta.json"
    assert meta_path.is_file()

    meta = json.loads(meta_path.read_text())
    assert meta["voice_id"] == "staged_voice"
    assert [item["filename"] for item in meta["files"]] == ["sample_a.WAV", "sample_b.wav"]
    assert [item["size"] for item in meta["files"]] == [128, 256]

    listed = store.list_all()
    assert len(listed) == 1
    assert listed[0].voice_id == "staged_voice"


def test_register_staged_voices_skips_existing_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "voices_dir", tmp_path)
    voice_dir = tmp_path / "pre_registered"
    voice_dir.mkdir(parents=True, exist_ok=True)
    (voice_dir / "sample.wav").write_bytes(b"\x00" * 64)

    original_meta = {
        "voice_id": "pre_registered",
        "created": 123,
        "model": "custom-model",
        "language": "fr",
        "files": [{"filename": "sample.wav", "size": 64}],
    }
    meta_path = voice_dir / "meta.json"
    meta_path.write_text(json.dumps(original_meta, indent=2))

    store = VoiceStore()
    registered = store.register_staged_voices()

    assert registered == 0
    assert json.loads(meta_path.read_text()) == original_meta
