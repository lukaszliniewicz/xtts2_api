import wave

from fastapi.testclient import TestClient
import pytest

from src.xtts_fastapi.main import app

client = TestClient(app)


def test_speech_invalid_model():
    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "nonexistent_model",
            "input": "Hello",
            "voice": "test_voice",
        },
    )
    assert resp.status_code in (400, 404)


def test_speech_missing_voice(monkeypatch):
    class FakeWrapper:
        speaker_manager = None

        def load(self):
            return None

    monkeypatch.setattr(
        "src.xtts_fastapi.main.engine._get_wrapper",
        lambda model_id, model_info=None: FakeWrapper(),
    )

    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "xtts_v2",
            "input": "Hello",
            "voice": "nonexistent_voice",
        },
    )

    assert resp.status_code in (400, 404)


def test_speech_unsupported_format():
    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "xtts_v2",
            "input": "Hello",
            "voice": "test",
            "response_format": "xyz",
        },
    )
    assert resp.status_code in (400, 422)


def test_speech_unsupported_language():
    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "xtts_v2",
            "input": "Hello",
            "voice": "test",
            "language": "xx",
        },
    )
    assert resp.status_code == 400
    data = resp.json()
    assert "unsupported_language" in str(data)


def test_speech_invalid_xtts_params():
    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "xtts_v2",
            "input": "Hello",
            "voice": "test",
            "xtts": {"temperature": 99.0},
        },
    )
    assert resp.status_code == 422


def test_speech_instructions_invalid_json():
    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "xtts_v2",
            "input": "Hello",
            "voice": "test",
            "instructions": "{not json",
        },
    )
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"]["code"] == "invalid_instructions_json"


def test_speech_instructions_invalid_xtts_params():
    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "xtts_v2",
            "input": "Hello",
            "voice": "test",
            "instructions": '{"xtts": {"temperature": 99.0}}',
        },
    )
    assert resp.status_code == 422


def test_speech_instructions_temp_alias(monkeypatch):
    captured = {}

    async def fake_generate(request, model_info=None):
        captured["temperature"] = request.xtts.temperature if request.xtts is not None else None
        return b"RIFF" + (b"\x00" * 64)

    monkeypatch.setattr("src.xtts_fastapi.main.engine.generate_speech_async", fake_generate)

    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "xtts_v2",
            "input": "Hello world",
            "voice": "alloy",
            "response_format": "wav",
            "instructions": '{"temp": 0.42}',
        },
    )
    assert resp.status_code == 200
    assert captured["temperature"] == pytest.approx(0.42)


def test_speech_instructions_max_ref_len_alias(monkeypatch):
    captured = {}

    async def fake_generate(request, model_info=None):
        captured["max_ref_length"] = request.xtts.max_ref_length if request.xtts is not None else None
        return b"RIFF" + (b"\x00" * 64)

    monkeypatch.setattr("src.xtts_fastapi.main.engine.generate_speech_async", fake_generate)

    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "xtts_v2",
            "input": "Hello world",
            "voice": "alloy",
            "response_format": "wav",
            "instructions": '{"max_ref_len": 9}',
        },
    )
    assert resp.status_code == 200
    assert captured["max_ref_length"] == 9


def test_speech_short_reference_audio_rejected(tmp_path):
    short_wav = tmp_path / "short.wav"
    with wave.open(str(short_wav), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 2400)

    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "xtts_v2",
            "input": "Hello",
            "voice": "test",
            "speaker_wav": [str(short_wav)],
        },
    )
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"]["code"] == "reference_audio_too_short"


def test_speech_schema_openai(monkeypatch):
    async def fake_generate(request, model_info=None):
        return b"RIFF" + (b"\x00" * 64)

    monkeypatch.setattr("src.xtts_fastapi.main.engine.generate_speech_async", fake_generate)

    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "xtts_v2",
            "input": "Hello world",
            "voice": "alloy",
            "response_format": "wav",
            "speed": 1.0,
        },
    )
    assert resp.status_code in (200, 400, 404)
