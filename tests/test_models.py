from fastapi.testclient import TestClient

from src.xtts_fastapi.main import app

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
