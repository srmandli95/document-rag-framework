"""Tests for text extraction with Day 20 authorization."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user
from app.db.database import get_db
from app.ingestion.loaders import extract_text_from_file
from app.main import app
from app.api import document_routes
from app.ingestion import extraction_service
from app.ingestion.cleaner import clean_text
from conftest import FakeDB, override_get_db, override_get_current_user


client = TestClient(app)


def setup_auth_overrides(user_id: str = "test-user"):
    """Setup both DB and auth overrides."""
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user(user_id)


@pytest.fixture(autouse=True)
def cleanup_overrides():
    """Cleanup dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


def test_extract_text_from_txt_loader():
    file_path = "sample_knowledge_base/sample_health_policy.txt"

    text = extract_text_from_file(
        file_path=file_path,
        content_type="text/plain",
        file_extension=".txt",
    )

    assert "Sample Health Policy" in text


def test_extract_text_from_md_loader():
    file_path = "sample_knowledge_base/sample_employee_benefits.md"

    text = extract_text_from_file(
        file_path=file_path,
        content_type="text/markdown",
        file_extension=".md",
    )

    assert "Sample Employee Benefits" in text


def test_extract_text_from_html_loader():
    file_path = "sample_knowledge_base/sample_internet_terms.html"

    text = extract_text_from_file(
        file_path=file_path,
        content_type="text/html",
        file_extension=".html",
    )

    assert "Sample Internet Service Terms" in text
    assert "ignore this script" not in text



def test_extract_endpoint_success(monkeypatch, tmp_path):
    """Extract endpoint returns extracted text for authenticated user."""
    setup_auth_overrides("test-user")

    fake_document = SimpleNamespace(
        id="fake-document-id",
        user_id="test-user",
        original_file_name="sample_health_policy.txt",
        stored_file_name="sample_health_policy.txt",
        category="health_insurance",
        content_type="text/plain",
        storage_path="sample_knowledge_base/sample_health_policy.txt",
        status="uploaded",
    )

    extracted_file = tmp_path / "extracted_text.txt"
    extracted_file.write_text("Sample Health Policy", encoding="utf-8")

    def fake_get_document_by_id(db, document_id: str, user_id: str):
        assert isinstance(db, FakeDB)
        assert document_id == "fake-document-id"
        assert user_id == "test-user"
        return fake_document

    def fake_extract_and_store_document_text(db, document):
        assert isinstance(db, FakeDB)
        assert document.id == "fake-document-id"

        return {
            "document_id": document.id,
            "user_id": document.user_id,
            "status": "extracted",
            "extracted_text_path": str(extracted_file),
            "character_count": len("Sample Health Policy"),
            "message": "Document text extracted successfully.",
        }

    monkeypatch.setattr(
        document_routes,
        "get_document_by_id",
        fake_get_document_by_id,
    )

    monkeypatch.setattr(
        document_routes,
        "extract_and_store_document_text",
        fake_extract_and_store_document_text,
    )

    response = client.post(
        "/documents/fake-document-id/extract"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "fake-document-id"
    assert data["user_id"] == "test-user"
    assert data["status"] == "extracted"


def test_extract_invalid_document_id_returns_404(monkeypatch):
    """Extract endpoint returns 404 for non-existent document."""
    setup_auth_overrides("test-user")

    def fake_get_document_by_id(db, document_id: str, user_id: str):
        assert isinstance(db, FakeDB)
        return None

    monkeypatch.setattr(
        document_routes,
        "get_document_by_id",
        fake_get_document_by_id,
    )

    response = client.post(
        "/documents/invalid-document-id/extract"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_extract_without_auth_returns_401():
    """Extract endpoint requires authentication."""
    # Don't setup overrides - test unauthenticated access
    response = client.post(
        "/documents/fake-document-id/extract"
    )

    assert response.status_code == 401


def test_clean_text_normalizes_spaces_and_blank_lines():
    raw_text = """
        Sample   Health     Policy


        Coverage     Summary:
            Primary care    visits are covered.



        Emergency care is covered.
    """

    cleaned = clean_text(raw_text)

    assert "Sample Health Policy" in cleaned
    assert "Coverage Summary:" in cleaned
    assert "Primary care visits are covered." in cleaned
    assert "Emergency care is covered." in cleaned

    # Should not keep excessive spaces
    assert "   " not in cleaned

    # Should not keep too many blank lines
    assert "\n\n\n" not in cleaned


def test_extract_and_store_document_text_writes_extracted_file(
    monkeypatch,
    tmp_path,
):
    """
    This test validates the real extraction service flow:

    1. Mark document as processing
    2. Extract text
    3. Clean text
    4. Write extracted_text.txt
    5. Mark document as extracted

    No real DB call happens because update_document_status is monkeypatched.
    """

    raw_file = tmp_path / "sample_health_policy.txt"
    raw_file.write_text(
        "Sample   Health    Policy\n\n\nPrimary care    visits are covered.",
        encoding="utf-8",
    )

    extracted_text_dir = tmp_path / "extracted_text"

    fake_document = SimpleNamespace(
        id="doc-123",
        user_id="user-123",
        content_type="text/plain",
        storage_path=str(raw_file),
        status="uploaded",
    )

    status_updates = []

    def fake_update_document_status(db, document_id: str, status: str):
        assert isinstance(db, FakeDB)
        assert document_id == "doc-123"

        status_updates.append(status)
        fake_document.status = status

        return fake_document

    monkeypatch.setattr(
        extraction_service.settings,
        "EXTRACTED_TEXT_DIR",
        str(extracted_text_dir),
    )

    monkeypatch.setattr(
        extraction_service,
        "update_document_status",
        fake_update_document_status,
    )

    result = extraction_service.extract_and_store_document_text(
        db=FakeDB(),
        document=fake_document,
    )

    assert result["document_id"] == "doc-123"
    assert result["user_id"] == "user-123"
    assert result["status"] == "extracted"
    assert result["character_count"] > 0
    assert result["message"] == "Document text extracted successfully."

    extracted_file = Path(result["extracted_text_path"])

    assert extracted_file.exists()
    assert extracted_file.name == "extracted_text.txt"

    extracted_content = extracted_file.read_text(encoding="utf-8")

    assert "Sample Health Policy" in extracted_content
    assert "Primary care visits are covered." in extracted_content

    # Confirms status flow
    assert status_updates == ["processing", "extracted"]


def test_extract_and_store_document_text_marks_failed_when_file_missing(
    monkeypatch,
    tmp_path,
):
    """
    This test validates failure behavior.

    If raw file is missing:
    1. status becomes processing
    2. service catches the failure
    3. status becomes failed
    4. response returns failed metadata

    No real DB call happens.
    """

    missing_file = tmp_path / "missing_file.txt"

    fake_document = SimpleNamespace(
        id="doc-failed",
        user_id="user-123",
        content_type="text/plain",
        storage_path=str(missing_file),
        status="uploaded",
    )

    status_updates = []

    def fake_update_document_status(db, document_id: str, status: str):
        assert isinstance(db, FakeDB)
        assert document_id == "doc-failed"

        status_updates.append(status)
        fake_document.status = status

        return fake_document

    monkeypatch.setattr(
        extraction_service.settings,
        "EXTRACTED_TEXT_DIR",
        str(tmp_path / "extracted_text"),
    )

    monkeypatch.setattr(
        extraction_service,
        "update_document_status",
        fake_update_document_status,
    )

    result = extraction_service.extract_and_store_document_text(
        db=FakeDB(),
        document=fake_document,
    )

    assert result["document_id"] == "doc-failed"
    assert result["user_id"] == "user-123"
    assert result["status"] == "failed"
    assert result["extracted_text_path"] == ""
    assert result["character_count"] == 0
    assert "failed" in result["message"].lower()

    assert status_updates == ["processing", "failed"]