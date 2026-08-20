import pytest
from fastapi.testclient import TestClient

from src.xtts_fastapi import main as main_module
from src.xtts_fastapi.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def xtts_runtime_available(monkeypatch):
    monkeypatch.setattr(main_module.model_loader, "HAS_XTTS", True)
    monkeypatch.setattr(main_module.model_loader, "XTTS_IMPORT_ERROR", None)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.3"


def test_health_methods():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_reports_unavailable_xtts_runtime(monkeypatch):
    monkeypatch.setattr(main_module.model_loader, "HAS_XTTS", False)
    monkeypatch.setattr(
        main_module.model_loader,
        "XTTS_IMPORT_ERROR",
        ImportError("Numba needs NumPy 2.4 or less"),
    )

    resp = client.get("/health")

    assert resp.status_code == 503
    assert resp.json()["status"] == "unavailable"
    assert "NumPy 2.4 or less" in resp.json()["runtime_error"]


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
