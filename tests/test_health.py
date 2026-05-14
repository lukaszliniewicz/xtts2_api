from fastapi.testclient import TestClient

from src.xtts_fastapi.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_health_methods():
    resp = client.get("/health")
    assert resp.status_code == 200
