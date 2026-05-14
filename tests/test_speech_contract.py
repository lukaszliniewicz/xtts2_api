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


def test_speech_missing_voice():
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
    assert resp.status_code == 400
    data = resp.json()
    assert "unsupported_format" in str(data)


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


def test_speech_schema_openai():
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
