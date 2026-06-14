from pathlib import Path
from uuid import uuid4
from fastapi.testclient import TestClient
from types import SimpleNamespace
from app.main import app

client = TestClient(app)






def test_upload_valid_text_file(monkeypatch):

    test_raw_dir = Path("/tmp/test_raw_documents")
    test_raw_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "app.config.settings.settings.RAW_DOCUMENTS_DIR",
        str(test_raw_dir),
    )

    def fake_create_document_record(
        db,
        *,
        user_id,
        original_file_name,
        stored_file_name,
        category,
        content_type,
        file_size_bytes,
        storage_provider,
        storage_path,
        status="uploaded",
    ):
        return SimpleNamespace(
            id=str(uuid4()),
            user_id=user_id,
            original_file_name=original_file_name,
            stored_file_name=stored_file_name,
            category=category,
            content_type=content_type,
            file_size_bytes=file_size_bytes,
            storage_provider=storage_provider,
            storage_path=storage_path,
            status=status,
            created_at=None,
            updated_at=None,
        )

    monkeypatch.setattr(
        "app.api.document_routes.create_document_record",
        fake_create_document_record,
    )
    monkeypatch.setattr(
        "app.api.document_routes.create_processing_job",
        lambda **kwargs: SimpleNamespace(
            id="job-1", status="pending", current_step=None, error_message=None
        ),
    )
    monkeypatch.setattr(
        "app.api.document_routes._process_document_in_background", lambda *args: None
    )

    file_content = "This is a synthetic test document for unit testing."

    response = client.post(
        "/documents/upload",
        data={
            "user_id": "test_user-123",
            "category": "test_category",
        },
        files={
            "file": (
                "test_document.txt",
                file_content,
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["document_id"] is not None
    assert body["user_id"] == "test_user-123"
    assert body["category"] == "test_category"
    assert body["storage_provider"] == "local"
    assert body["status"] == "uploaded"
    assert body["file_name"] == "test_document.txt"
    assert body["original_file_name"] == "test_document.txt"
    assert body["content_type"] == "text/plain"
    assert body["file_size_bytes"] == len(file_content)
    assert body["job_id"] == "job-1"
    assert body["message"] == "Document uploaded and processing started"

    saved_file_path = Path(body["storage_path"])
    assert saved_file_path.exists()
    assert "tmp/test_raw_documents" in str(saved_file_path)

def test_unsupported_file_extension():
    response = client.post(
        "/documents/upload",
        data={
            "user_id": "test_user-123",
            "category": "test_category",
        },
        files={
            "file": (
                "test_document.exe",
                b"fake content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]

def test_reject_unsupported_content_type():
    response = client.post(
        "/documents/upload",
        data={
            "user_id": "test_user-123",
            "category": "test_category",
        },
        files={
            "file": (
                "test_document.txt",
                b"fake content",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400
    assert "Unsupported content type" in response.json()["detail"]

def test_reject_missing_category():
    response = client.post(
        "/documents/upload",
        data={
            "user_id": "test-user-123",
        },
        files={
            "file": (
                "sample.txt",
                b"test content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 422

def test_reject_unauthenticated_upload():
    response = client.post(
        "/documents/upload",
        data={
            "category": "health_insurance",
        },
        files={
            "file": (
                "sample.txt",
                b"test content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 401


def test_reject_file_larger_than_max_size(monkeypatch):
    max_size_mb = 1
    monkeypatch.setattr(
        "app.config.settings.settings.MAX_UPLOAD_SIZE_MB",
        max_size_mb,
    )

    large_content = b"a" * (max_size_mb * 1024 * 1024 + 1)

    response = client.post(
        "/documents/upload",
        data={
            "user_id": "test_user-123",
            "category": "test_category",
        },
        files={
            "file": (
                "large_file.txt",
                large_content,
                "text/plain",
            )
        },
    )

    assert response.status_code == 413
    assert f"File size exceeds {max_size_mb} MB limit" in response.json()["detail"]
