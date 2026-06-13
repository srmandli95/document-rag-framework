from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth.dependencies import get_current_user
from app.db.database import get_db
from app.main import app
from app.models.document import Document


@pytest.fixture
def document_filter_client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Document.__table__.create(engine)
    testing_session = sessionmaker(bind=engine)
    db = testing_session()

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="user-a")

    client = TestClient(app)
    yield client, db

    client.close()
    db.close()
    Document.__table__.drop(engine)


def add_document(
    db,
    *,
    document_id: str,
    user_id: str = "user-a",
    file_name: str = "policy.txt",
    category: str = "general",
    status: str = "uploaded",
    created_at: datetime | None = None,
) -> Document:
    document = Document(
        id=document_id,
        user_id=user_id,
        original_file_name=file_name,
        stored_file_name=file_name,
        category=category,
        content_type="text/plain",
        file_size_bytes=100,
        storage_provider="local",
        storage_path=f"/tmp/{document_id}/{file_name}",
        status=status,
        created_at=created_at or datetime.utcnow(),
        updated_at=created_at or datetime.utcnow(),
    )
    db.add(document)
    db.commit()
    return document


def document_ids(response) -> list[str]:
    assert response.status_code == 200
    return [document["document_id"] for document in response.json()["documents"]]


def test_get_documents_returns_all_non_deleted_user_documents(document_filter_client):
    client, db = document_filter_client
    older = datetime.utcnow() - timedelta(days=1)
    add_document(db, document_id="older", created_at=older)
    add_document(db, document_id="newer", status="embedded")
    add_document(db, document_id="deleted", status="deleted")

    assert document_ids(client.get("/documents")) == ["newer", "older"]


def test_status_filter_returns_matching_documents(document_filter_client):
    client, db = document_filter_client
    add_document(db, document_id="uploaded", status="uploaded")
    add_document(db, document_id="embedded", status="embedded")

    assert document_ids(client.get("/documents?status=embedded")) == ["embedded"]


def test_category_filter_returns_matching_documents(document_filter_client):
    client, db = document_filter_client
    add_document(db, document_id="health", category="health_insurance")
    add_document(db, document_id="utility", category="utility")

    assert document_ids(
        client.get("/documents?category=health_insurance")
    ) == ["health"]


def test_search_filter_is_case_insensitive(document_filter_client):
    client, db = document_filter_client
    add_document(db, document_id="dte", file_name="DTE_Energy_Guide.txt")
    add_document(db, document_id="other", file_name="Health_Policy.txt")

    assert document_ids(client.get("/documents?search=dte")) == ["dte"]


def test_ready_only_returns_only_embedded_documents(document_filter_client):
    client, db = document_filter_client
    add_document(db, document_id="uploaded", status="uploaded")
    add_document(db, document_id="embedded", status="embedded")

    assert document_ids(client.get("/documents?ready_only=true")) == ["embedded"]


def test_combined_filters_apply_together(document_filter_client):
    client, db = document_filter_client
    add_document(
        db,
        document_id="match",
        file_name="DTE Guide.txt",
        category="utility",
        status="embedded",
    )
    add_document(
        db,
        document_id="wrong-status",
        file_name="DTE Bill.txt",
        category="utility",
        status="uploaded",
    )
    add_document(
        db,
        document_id="wrong-category",
        file_name="DTE Benefits.txt",
        category="employer_benefits",
        status="embedded",
    )

    assert document_ids(
        client.get("/documents?category=utility&status=embedded&search=dte")
    ) == ["match"]


def test_invalid_status_returns_400(document_filter_client):
    client, _ = document_filter_client

    response = client.get("/documents?status=deleted")

    assert response.status_code == 400
    assert "Unsupported document status" in response.json()["detail"]


def test_invalid_category_returns_400(document_filter_client):
    client, _ = document_filter_client

    response = client.get("/documents?category=not_real")

    assert response.status_code == 400
    assert "Unsupported document category" in response.json()["detail"]


def test_deleted_documents_are_hidden_even_when_search_matches(document_filter_client):
    client, db = document_filter_client
    add_document(db, document_id="deleted", file_name="DTE.txt", status="deleted")

    assert document_ids(client.get("/documents?search=dte")) == []


def test_user_cannot_see_another_users_documents(document_filter_client):
    client, db = document_filter_client
    add_document(db, document_id="owner", user_id="user-a")
    add_document(db, document_id="other", user_id="user-b")

    assert document_ids(client.get("/documents")) == ["owner"]
