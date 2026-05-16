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


def test_health_sets_request_id_header():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "x-request-id" in resp.headers
    assert resp.headers["x-request-id"]


def test_health_reuses_supplied_request_id():
    request_id = "req-test-123"
    resp = client.get("/health", headers={"X-Request-ID": request_id})
    assert resp.status_code == 200
    assert resp.headers["x-request-id"] == request_id
