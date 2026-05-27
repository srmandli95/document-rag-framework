from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_upload_valid_text_file(monkeypatch):

    test_raw_dir = Path("/tmp/test_raw_documents")
    test_raw_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "app.config.settings.settings.RAW_DOCUMENTS_DIR",
        str(test_raw_dir),
    )

    file_content = "This is a sythetic test document for unit testing."

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

    assert body["user_id"] == "test_user-123"
    assert body["category"] == "test_category"
    assert body["storage_provider"] == "local"
    assert body["status"] == "uploaded"
    assert body["file_name"] == "test_document.txt"
    assert body["file_size_bytes"] == len(file_content)

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

def test_reject_missing_user_id():
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

    assert response.status_code == 422


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

    assert response.status_code == 400
    assert f"File size exceeds the maximum allowed size of {max_size_mb} MB" in response.json()["detail"]