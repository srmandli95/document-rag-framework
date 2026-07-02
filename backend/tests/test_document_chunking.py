from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.db.database import get_db
from app.ingestion.chunker import chunk_text, parse_document_structure
from app.ingestion.chunking_service import chunk_and_store_document_text
from app.main import app
from app.api import document_routes


client = TestClient(app)


class FakeDB:
    pass


@pytest.fixture(autouse=True)
def override_db_dependency():
    def fake_get_db():
        yield FakeDB()

    app.dependency_overrides[get_db] = fake_get_db

    yield

    app.dependency_overrides.clear()


def test_chunk_text_returns_one_chunk_for_short_text():
    text = "This is a short health policy document."

    chunks = chunk_text(
        text=text,
        chunk_size=700,
        chunk_overlap=100,
    )

    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["token_count"] == 7
    assert chunks[0]["page_number"] is None
    assert chunks[0]["summary"]
    assert chunks[0]["keywords"]
    assert chunks[0]["hypothetical_questions"]
    assert chunks[0]["search_text"].startswith(chunks[0]["chunk_text"])


def test_chunk_text_returns_multiple_chunks_for_long_text():
    text = " ".join([f"word{i}" for i in range(1500)])

    chunks = chunk_text(
        text=text,
        chunk_size=700,
        chunk_overlap=100,
    )

    assert len(chunks) > 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1


def test_chunk_text_includes_overlap():
    text = " ".join([f"word{i}" for i in range(1000)])

    chunks = chunk_text(
        text=text,
        chunk_size=100,
        chunk_overlap=20,
    )

    first_chunk_words = chunks[0]["chunk_text"].split()
    second_chunk_words = chunks[1]["chunk_text"].split()

    assert first_chunk_words[-20:] == second_chunk_words[:20]


def test_chunk_text_rejects_overlap_greater_than_or_equal_to_chunk_size():
    with pytest.raises(ValueError):
        chunk_text(
            text="sample text",
            chunk_size=100,
            chunk_overlap=100,
        )


def test_chunk_text_empty_text_returns_empty_list():
    chunks = chunk_text("   ")

    assert chunks == []


def test_parse_document_structure_detects_headings_tables_and_pages():
    text = """--- Page 2 ---
# Coverage Rules

Urgent care is covered after a copay.

Plan | Copay | Limit
Urgent Care | $40 | 5 visits
Emergency | $250 | Unlimited
"""

    blocks = parse_document_structure(text)

    assert [block["type"] for block in blocks] == ["heading", "paragraph", "table"]
    assert blocks[0]["section_title"] == "Coverage Rules"
    assert blocks[1]["page_number"] == 2
    assert "Urgent Care | $40 | 5 visits" in blocks[2]["text"]


def test_chunk_text_preserves_table_and_attaches_enriched_metadata():
    text = """Coverage Rules

Urgent care is covered after a $40 copay.

Service | Copay | Notes
Urgent care | $40 | In network
Emergency room | $250 | Waived if admitted

Claims

Claims must be submitted within 90 days.
"""

    chunks = chunk_text(text=text, chunk_size=80, chunk_overlap=10)

    coverage_chunk = chunks[0]
    claims_chunk = chunks[1]

    assert coverage_chunk["section_title"] == "Coverage Rules"
    assert "Service | Copay | Notes" in coverage_chunk["chunk_text"]
    assert "table" in coverage_chunk["structure_types"]
    assert coverage_chunk["summary"].startswith("Coverage Rules")
    assert "Questions:" in coverage_chunk["search_text"]
    assert coverage_chunk["hypothetical_questions"][0] == (
        "What does the document say about Coverage Rules?"
    )
    assert claims_chunk["section_title"] == "Claims"


def test_chunking_endpoint_success_without_real_db_calls(monkeypatch):
    fake_document = SimpleNamespace(
        id="doc-123",
        user_id="user-123",
        status="extracted",
    )

    def fake_get_document_by_id(db, document_id, user_id):
        return fake_document

    def fake_chunk_and_store_document_text(db, document):
        return {
            "document_id": document.id,
            "user_id": document.user_id,
            "status": "chunked",
            "chunk_count": 3,
            "message": "Document text chunked successfully",
        }

    monkeypatch.setattr(
        document_routes,
        "get_document_by_id",
        fake_get_document_by_id,
    )

    monkeypatch.setattr(
        document_routes,
        "chunk_and_store_document_text",
        fake_chunk_and_store_document_text,
    )

    response = client.post(
        "/documents/doc-123/chunk?user_id=user-123"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["document_id"] == "doc-123"
    assert data["user_id"] == "user-123"
    assert data["status"] == "chunked"
    assert data["chunk_count"] == 3


def test_chunking_endpoint_invalid_document_returns_404(monkeypatch):
    def fake_get_document_by_id(db, document_id, user_id):
        return None

    monkeypatch.setattr(
        document_routes,
        "get_document_by_id",
        fake_get_document_by_id,
    )

    response = client.post(
        "/documents/missing-doc/chunk?user_id=user-123"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_user_a_cannot_chunk_user_b_document(monkeypatch):
    def fake_get_document_by_id(db, document_id, user_id):
        return None

    monkeypatch.setattr(
        document_routes,
        "get_document_by_id",
        fake_get_document_by_id,
    )

    response = client.post(
        "/documents/doc-owned-by-user-b/chunk?user_id=user-a"
    )

    assert response.status_code == 404


def test_document_not_extracted_returns_400(monkeypatch):
    fake_document = SimpleNamespace(
        id="doc-123",
        user_id="user-123",
        status="uploaded",
    )

    def fake_get_document_by_id(db, document_id, user_id):
        return fake_document

    monkeypatch.setattr(
        document_routes,
        "get_document_by_id",
        fake_get_document_by_id,
    )

    response = client.post(
        "/documents/doc-123/chunk?user_id=user-123"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Document must be extracted before chunking"


def test_chunking_service_reads_text_and_writes_chunks(monkeypatch, tmp_path):
    extracted_base_dir = tmp_path / "extracted_text"
    user_id = "user-123"
    document_id = "doc-123"

    extracted_dir = extracted_base_dir / user_id / document_id
    extracted_dir.mkdir(parents=True)

    extracted_file = extracted_dir / "extracted_text.txt"
    extracted_file.write_text(
        " ".join(["sample"] * 1000),
        encoding="utf-8",
    )

    fake_document = SimpleNamespace(
        id=document_id,
        user_id=user_id,
        status="extracted",
    )

    status_updates = []
    created_chunks = []

    def fake_update_document_status(db, document_id, status):
        status_updates.append(status)

    def fake_delete_chunks_by_document(db, document_id, user_id):
        return 0

    def fake_create_document_chunks(db, document_id, user_id, chunks):
        created_chunks.extend(chunks)
        return chunks

    monkeypatch.setattr(
        "app.ingestion.chunking_service.settings.EXTRACTED_TEXT_DIR",
        str(extracted_base_dir),
    )

    monkeypatch.setattr(
        "app.ingestion.chunking_service.update_document_status",
        fake_update_document_status,
    )

    monkeypatch.setattr(
        "app.ingestion.chunking_service.delete_chunks_by_document",
        fake_delete_chunks_by_document,
    )

    monkeypatch.setattr(
        "app.ingestion.chunking_service.create_document_chunks",
        fake_create_document_chunks,
    )

    result = chunk_and_store_document_text(
        db=FakeDB(),
        document=fake_document,
    )

    assert result["document_id"] == document_id
    assert result["user_id"] == user_id
    assert result["status"] == "chunked"
    assert result["chunk_count"] > 0
    assert "processing" in status_updates
    assert "chunked" in status_updates
    assert len(created_chunks) > 0
    assert created_chunks[0]["summary"]
    assert created_chunks[0]["keywords"]
    assert created_chunks[0]["hypothetical_questions"]
    assert created_chunks[0]["search_text"].startswith(created_chunks[0]["chunk_text"])


def test_failed_chunking_marks_document_status_failed(monkeypatch, tmp_path):
    extracted_base_dir = tmp_path / "extracted_text"
    user_id = "user-123"
    document_id = "doc-123"

    extracted_dir = extracted_base_dir / user_id / document_id
    extracted_dir.mkdir(parents=True)

    extracted_file = extracted_dir / "extracted_text.txt"
    extracted_file.write_text("sample text", encoding="utf-8")

    fake_document = SimpleNamespace(
        id=document_id,
        user_id=user_id,
        status="extracted",
    )

    status_updates = []

    def fake_update_document_status(db, document_id, status):
        status_updates.append(status)

    def fake_delete_chunks_by_document(db, document_id, user_id):
        raise RuntimeError("DB error while deleting chunks")

    monkeypatch.setattr(
        "app.ingestion.chunking_service.settings.EXTRACTED_TEXT_DIR",
        str(extracted_base_dir),
    )

    monkeypatch.setattr(
        "app.ingestion.chunking_service.update_document_status",
        fake_update_document_status,
    )

    monkeypatch.setattr(
        "app.ingestion.chunking_service.delete_chunks_by_document",
        fake_delete_chunks_by_document,
    )

    with pytest.raises(RuntimeError):
        chunk_and_store_document_text(
            db=FakeDB(),
            document=fake_document,
        )

    assert "processing" in status_updates
    assert "failed" in status_updates
