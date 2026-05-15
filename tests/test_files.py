from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from src.xtts_fastapi.engine import engine
from src.xtts_fastapi.api_models import CreateSpeechRequest
from src.xtts_fastapi.main import app

client = TestClient(app)


def test_files_crud_roundtrip():
    payload = b"RIFF" + (b"\x00" * 1024)

    created = client.post(
        "/v1/files",
        files={"file": ("sample_voice.wav", payload, "audio/wav")},
        data={"purpose": "user_data"},
    )
    assert created.status_code == 200
    file_obj = created.json()
    assert file_obj["object"] == "file"
    assert file_obj["filename"] == "sample_voice.wav"
    assert file_obj["bytes"] == len(payload)
    file_id = file_obj["id"]

    listed = client.get("/v1/files")
    assert listed.status_code == 200
    listed_data = listed.json()
    assert listed_data["object"] == "list"
    assert any(item["id"] == file_id for item in listed_data["data"])

    retrieved = client.get(f"/v1/files/{file_id}")
    assert retrieved.status_code == 200
    assert retrieved.json()["id"] == file_id

    content = client.get(f"/v1/files/{file_id}/content")
    assert content.status_code == 200
    assert content.content == payload

    deleted = client.delete(f"/v1/files/{file_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"id": file_id, "object": "file", "deleted": True}

    missing = client.get(f"/v1/files/{file_id}")
    assert missing.status_code == 404


def test_files_list_filters_and_cursor():
    purpose = f"voice_test_{uuid4().hex}"
    payload_a = b"A" * 256
    payload_b = b"B" * 512
    ids: list[str] = []

    try:
        created_a = client.post(
            "/v1/files",
            files={"file": ("a.wav", payload_a, "audio/wav")},
            data={"purpose": purpose},
        )
        assert created_a.status_code == 200
        ids.append(created_a.json()["id"])

        created_b = client.post(
            "/v1/files",
            files={"file": ("b.wav", payload_b, "audio/wav")},
            data={"purpose": purpose},
        )
        assert created_b.status_code == 200
        ids.append(created_b.json()["id"])

        filtered = client.get(f"/v1/files?purpose={purpose}&order=asc")
        assert filtered.status_code == 200
        filtered_data = filtered.json()["data"]
        filtered_ids = [item["id"] for item in filtered_data]
        for file_id in ids:
            assert file_id in filtered_ids

        first_page = client.get(f"/v1/files?purpose={purpose}&order=asc&limit=1")
        assert first_page.status_code == 200
        first_page_data = first_page.json()
        assert len(first_page_data["data"]) == 1
        assert first_page_data["has_more"] is True

        first_id = first_page_data["data"][0]["id"]
        second_page = client.get(f"/v1/files?purpose={purpose}&order=asc&after={first_id}")
        assert second_page.status_code == 200
        second_ids = [item["id"] for item in second_page.json()["data"]]
        assert first_id not in second_ids
        assert any(file_id in second_ids for file_id in ids if file_id != first_id)
    finally:
        for file_id in ids:
            client.delete(f"/v1/files/{file_id}")


def test_engine_resolves_file_id_to_local_path():
    payload = b"RIFF" + (b"\x00" * 512)

    created = client.post(
        "/v1/files",
        files={"file": ("voice_for_engine.wav", payload, "audio/wav")},
        data={"purpose": "user_data"},
    )
    assert created.status_code == 200
    file_id = created.json()["id"]

    try:
        request = CreateSpeechRequest(model="xtts_v2", input="Hello", voice=file_id)
        _, speaker_wav_paths = engine._resolve_voice(request)
        assert speaker_wav_paths is not None
        assert len(speaker_wav_paths) == 1
        assert Path(speaker_wav_paths[0]).is_file()
    finally:
        client.delete(f"/v1/files/{file_id}")
