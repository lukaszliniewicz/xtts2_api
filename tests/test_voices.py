from fastapi.testclient import TestClient

from src.xtts_fastapi.main import app

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
