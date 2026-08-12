from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from src.xtts_fastapi.errors import APIError
from src.xtts_fastapi.main import app
from src.xtts_fastapi.model_uploads import install_model_upload
from src.xtts_fastapi.registry import registry
from src.xtts_fastapi.settings import settings

client = TestClient(app, client=("127.0.0.1", 50000))


@pytest.fixture
def models_root(tmp_path, monkeypatch):
    original_models_dir = settings.models_dir
    root = tmp_path / "models"
    monkeypatch.setattr(settings, "models_dir", root)
    registry.refresh()
    try:
        yield root
    finally:
        monkeypatch.setattr(settings, "models_dir", original_models_dir)
        registry.refresh()


def _bundle_files(
    *,
    config: bytes = b'{"model":"xtts"}',
    model: bytes = b"model-weights",
    speakers: bytes = b"speaker-data",
    vocab: bytes = b'{"en":0}',
) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [
        ("files", ("config.json", config, "application/json")),
        ("files", ("model.pth", model, "application/octet-stream")),
        ("files", ("speakers_xtts.pth", speakers, "application/octet-stream")),
        ("files", ("vocab.json", vocab, "application/json")),
    ]


def _post_model(model_id: str, files: list[tuple[str, tuple[str, bytes, str]]]):
    return client.post("/v1/models", data={"model_id": model_id}, files=files)


def test_upload_model_installs_complete_bundle_and_lists_it(models_root):
    files = _bundle_files()
    expected_bytes = sum(len(item[1][1]) for item in files)

    response = _post_model("acme-voice", files)

    assert response.status_code == 201
    assert response.json() == {
        "id": "acme-voice",
        "object": "model",
        "created": 0,
        "owned_by": "xtts-fapi",
        "bytes": expected_bytes,
    }
    assert {path.name for path in (models_root / "acme-voice").iterdir()} == {
        "config.json",
        "model.pth",
        "speakers_xtts.pth",
        "vocab.json",
    }

    listed = client.get("/v1/models")
    assert listed.status_code == 200
    assert "acme-voice" in {model["id"] for model in listed.json()["data"]}


def test_upload_model_allows_nested_relative_id(models_root):
    response = _post_model("team-a/english/v1", _bundle_files())

    assert response.status_code == 201
    assert response.json()["id"] == "team-a/english/v1"
    assert (models_root / "team-a" / "english" / "v1" / "model.pth").is_file()


@pytest.mark.parametrize("model_id", ["release.v1", "team.alpha/voice.v2"])
def test_upload_model_allows_ordinary_dotted_ids(models_root, model_id):
    response = _post_model(model_id, _bundle_files())

    assert response.status_code == 201
    assert response.json()["id"] == model_id


def test_upload_model_reports_missing_form_fields_as_api_errors(models_root):
    missing_model_id = client.post("/v1/models", files=_bundle_files())
    missing_files = client.post("/v1/models", data={"model_id": "missing-files"})

    assert missing_model_id.status_code == 400
    assert missing_model_id.json()["error"]["code"] == "missing_model_id"
    assert missing_files.status_code == 400
    assert missing_files.json()["error"]["code"] == "missing_model_files"


def test_upload_model_rejects_non_loopback_client_before_parsing(models_root):
    remote_client = TestClient(app, client=("192.0.2.10", 50000))

    response = remote_client.post(
        "/v1/models",
        data={"model_id": "remote"},
        files=_bundle_files(),
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "model_upload_loopback_only"
    assert not (models_root / "remote").exists()


def test_upload_model_requires_content_length_before_parsing(models_root):
    response = client.post(
        "/v1/models",
        content=iter([b"unframed multipart body"]),
        headers={"Content-Type": "multipart/form-data; boundary=example"},
    )

    assert response.status_code == 411
    assert response.json()["error"]["code"] == "model_upload_length_required"


def test_upload_model_rejects_oversized_content_length_before_parsing(models_root):
    response = client.post(
        "/v1/models",
        content=b"",
        headers={
            "Content-Type": "multipart/form-data; boundary=example",
            "Content-Length": str(settings.model_upload_max_total_bytes + 16 * 1024 * 1024 + 1),
        },
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "model_upload_too_large"


@pytest.mark.parametrize(
    ("files", "code"),
    [
        (_bundle_files()[:-1], "missing_model_files"),
        (_bundle_files() + [("files", ("notes.txt", b"notes", "text/plain"))], "unexpected_model_file"),
        (_bundle_files() + [("files", ("config.json", b'{"model":"xtts"}', "application/json"))], "duplicate_model_file"),
        (_bundle_files(model=b""), "empty_model_file"),
    ],
)
def test_upload_model_rejects_invalid_bundle_contents(models_root, files, code):
    response = _post_model("invalid-bundle", files)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == code
    assert not (models_root / "invalid-bundle").exists()
    assert not any((models_root / ".downloads").iterdir())


@pytest.mark.parametrize(
    "config",
    [b"not-json", b"[]", b"{}", b'{"model":"other"}'],
)
def test_upload_model_rejects_invalid_xtts_config(models_root, config):
    response = _post_model("invalid-config", _bundle_files(config=config))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_model_config"
    assert not (models_root / "invalid-config").exists()
    assert not any((models_root / ".downloads").iterdir())


@pytest.mark.parametrize(
    "model_id",
    [
        "../escape",
        "/absolute",
        ".hidden",
        "nested/.hidden",
        r"nested\\escape",
        "trailing-dot.",
        "nested/trailing-space ",
        "con",
        "CON.txt",
        "lpt1.model",
        "nested/AUX.json",
    ],
)
def test_upload_model_rejects_unsafe_model_ids(models_root, model_id):
    response = _post_model(model_id, _bundle_files())

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_model_id"
    downloads_dir = models_root / ".downloads"
    assert not downloads_dir.exists() or not any(downloads_dir.iterdir())


def test_upload_model_rejects_symlink_escape(models_root):
    models_root.mkdir()
    outside = models_root.parent / "outside"
    outside.mkdir()
    (models_root / "linked").symlink_to(outside, target_is_directory=True)

    response = _post_model("linked/escape", _bundle_files())

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_model_id"
    assert not (outside / "escape").exists()


def test_upload_model_rejects_existing_target_without_overwriting(models_root):
    target = models_root / "already-there"
    target.mkdir(parents=True)
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    response = _post_model("already-there", _bundle_files())

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "model_already_exists"
    assert marker.read_text(encoding="utf-8") == "keep"


def test_upload_model_rejects_file_size_limit(models_root, monkeypatch):
    monkeypatch.setattr(settings, "model_upload_max_file_bytes", 8)
    monkeypatch.setattr(settings, "model_upload_max_config_bytes", 100)

    response = _post_model("too-large", _bundle_files(model=b"x" * 9))

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "model_upload_too_large"
    assert not (models_root / "too-large").exists()
    assert not any((models_root / ".downloads").iterdir())


def test_upload_model_rejects_total_size_limit(models_root, monkeypatch):
    monkeypatch.setattr(settings, "model_upload_max_total_bytes", 30)

    response = _post_model(
        "total-too-large",
        _bundle_files(model=b"m" * 10, speakers=b"s" * 10, vocab=b"v" * 10),
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "model_upload_too_large"
    assert not (models_root / "total-too-large").exists()
    assert not any((models_root / ".downloads").iterdir())


class _MemoryUpload:
    def __init__(self, filename: str, data: bytes, *, fail_after_reads: int | None = None):
        self.filename = filename
        self._data = data
        self._position = 0
        self._reads = 0
        self._fail_after_reads = fail_after_reads
        self.read_sizes: list[int] = []
        self.closed = False

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if self._fail_after_reads is not None and self._reads >= self._fail_after_reads:
            raise OSError("simulated interrupted upload")
        self._reads += 1
        if size < 0:
            return self._data[self._position :]
        chunk = self._data[self._position : self._position + size]
        self._position += len(chunk)
        return chunk

    async def close(self) -> None:
        self.closed = True


def _memory_bundle(*, model: bytes = b"model", fail_model_after_reads: int | None = None) -> list[_MemoryUpload]:
    return [
        _MemoryUpload("config.json", b'{"model":"xtts"}'),
        _MemoryUpload("model.pth", model, fail_after_reads=fail_model_after_reads),
        _MemoryUpload("speakers_xtts.pth", b"speakers"),
        _MemoryUpload("vocab.json", b'{"en":0}'),
    ]


def test_interrupted_copy_cleans_staging_and_never_publishes_partial_model(models_root):
    uploads = _memory_bundle(fail_model_after_reads=1)

    with pytest.raises(APIError) as error:
        asyncio.run(install_model_upload("interrupted", uploads))

    assert error.value.code == "model_upload_failed"
    assert not (models_root / "interrupted").exists()
    assert registry.get("interrupted") is None
    assert not any((models_root / ".downloads").iterdir())
    assert all(upload.closed for upload in uploads)


def test_model_upload_reads_in_bounded_chunks(models_root, monkeypatch):
    chunk_size = 64 * 1024
    monkeypatch.setattr(settings, "model_upload_chunk_size", chunk_size)
    model_data = b"m" * (chunk_size * 3 + 17)
    uploads = _memory_bundle(model=model_data)

    response = asyncio.run(install_model_upload("streamed", uploads))

    model_upload = next(upload for upload in uploads if upload.filename == "model.pth")
    assert response.id == "streamed"
    assert len(model_upload.read_sizes) >= 5
    assert set(model_upload.read_sizes) == {chunk_size}
    assert all(upload.closed for upload in uploads)
