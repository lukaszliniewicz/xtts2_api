from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from src.xtts_fastapi.api_models import CreateSpeechRequest
from src.xtts_fastapi.engine import engine
from src.xtts_fastapi.file_store import file_store
from src.xtts_fastapi.main import app
from src.xtts_fastapi.voices import normalize_voice_id

client = TestClient(app)


def test_files_create_rejects_non_wav_upload():
    payload = b"plain-text-payload"

    created = client.post(
        "/v1/files",
        files={"file": ("sample.txt", payload, "text/plain")},
        data={"purpose": "user_data"},
    )
    assert created.status_code == 422
    data = created.json()
    assert data["error"]["code"] == "unsupported_file_type"


def test_files_wav_upload_promotes_to_voice():
    payload = b"RIFF\x00\x00\x00\x00WAVE" + (b"\x00" * 1024)
    source_name = f"My Cool Voice ({uuid4().hex}).wav"
    expected_voice_id = normalize_voice_id(Path(source_name).stem)

    created = client.post(
        "/v1/files",
        files={"file": (source_name, payload, "audio/wav")},
        data={"purpose": "user_data"},
    )
    assert created.status_code == 200
    voice_obj = created.json()
    assert voice_obj["object"] == "voice"
    assert voice_obj["id"] == expected_voice_id
    assert voice_obj["sample_count"] == 1

    listed_voices = client.get("/v1/voices")
    assert listed_voices.status_code == 200
    voice_ids = [item["voice_id"] for item in listed_voices.json()["data"]]
    assert expected_voice_id in voice_ids

    listed_files = client.get("/v1/files")
    assert listed_files.status_code == 200
    file_ids = [item["id"] for item in listed_files.json()["data"]]
    assert expected_voice_id not in file_ids

    sample_path = Path("voices") / expected_voice_id / f"{expected_voice_id}.wav"
    assert sample_path.is_file()

    client.delete(f"/v1/voices/{expected_voice_id}")


def test_files_wav_upload_uses_name_override_for_voice_id():
    payload = b"RIFF\x00\x00\x00\x00WAVE" + (b"\x00" * 1024)
    override_name = f"Team Voice #{uuid4().hex}.wav"
    expected_voice_id = normalize_voice_id(Path(override_name).stem)

    created = client.post(
        "/v1/files",
        files={"file": ("upload_tmp.wav", payload, "audio/wav")},
        data={"purpose": "user_data", "name": override_name},
    )
    assert created.status_code == 200
    voice_obj = created.json()
    assert voice_obj["id"] == expected_voice_id

    sample_path = Path("voices") / expected_voice_id / f"{expected_voice_id}.wav"
    assert sample_path.is_file()

    client.delete(f"/v1/voices/{expected_voice_id}")


def test_files_create_rejects_empty_upload():
    created = client.post(
        "/v1/files",
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"purpose": "user_data"},
    )
    assert created.status_code == 400
    data = created.json()
    assert data["error"]["code"] == "empty_file"


def test_legacy_files_crud_roundtrip():
    payload = b"legacy-content"
    file_obj = file_store.create(
        filename=f"legacy-{uuid4().hex}.bin",
        data=payload,
        purpose="user_data",
    )
    file_id = file_obj.id

    try:
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
    finally:
        client.delete(f"/v1/files/{file_id}")


def test_files_list_filters_and_cursor_for_legacy_entries():
    purpose = f"legacy_voice_test_{uuid4().hex}"
    ids: list[str] = []

    try:
        created_a = file_store.create(filename=f"a-{uuid4().hex}.bin", data=b"A" * 256, purpose=purpose)
        ids.append(created_a.id)

        created_b = file_store.create(filename=f"b-{uuid4().hex}.bin", data=b"B" * 512, purpose=purpose)
        ids.append(created_b.id)

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


def test_engine_resolves_voice_id_from_files_wav_upload():
    payload = b"RIFF\x00\x00\x00\x00WAVE" + (b"\x00" * 512)
    source_name = f"voice_for_engine_{uuid4().hex}.wav"
    expected_voice_id = normalize_voice_id(Path(source_name).stem)

    created = client.post(
        "/v1/files",
        files={"file": (source_name, payload, "audio/wav")},
        data={"purpose": "user_data"},
    )
    assert created.status_code == 200
    voice_id = created.json()["id"]
    assert voice_id == expected_voice_id

    try:
        request = CreateSpeechRequest(model="xtts_v2", input="Hello", voice=voice_id)
        _, speaker_wav_paths = engine._resolve_voice(request)
        assert speaker_wav_paths is not None
        assert len(speaker_wav_paths) == 1
        assert Path(speaker_wav_paths[0]).is_file()
    finally:
        client.delete(f"/v1/voices/{voice_id}")
