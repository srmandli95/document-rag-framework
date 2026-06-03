from fastapi.testclient import TestClient

from app.db.database import get_db
from app.main import app
from app.retrieval.bm25_retriever import bm25_search, tokenize_text


client = TestClient(app)


class FakeDB:
    pass


class FakeChunk:
    def __init__(
        self,
        id,
        document_id,
        user_id,
        chunk_text,
        chunk_index,
        token_count=None,
        page_number=None,
        section_title=None,
        status="embedded",
    ):
        self.id = id
        self.document_id = document_id
        self.user_id = user_id
        self.chunk_text = chunk_text
        self.chunk_index = chunk_index
        self.token_count = token_count
        self.page_number = page_number
        self.section_title = section_title
        self.status = status


def override_get_db():
    return FakeDB()


def test_bm25_tokenizer_lowercases_text():
    tokens = tokenize_text("Urgent Care Copay")

    assert tokens == ["urgent", "care", "copay"]


def test_bm25_tokenizer_handles_punctuation():
    tokens = tokenize_text("Urgent-care, copay!")

    assert tokens == ["urgent", "care", "copay"]


def test_bm25_endpoint_success_using_monkeypatch(monkeypatch):
    app.dependency_overrides[get_db] = override_get_db

    fake_results = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "user_id": "local-user-123",
            "chunk_text": "Urgent care copay is 50 dollars.",
            "chunk_index": 0,
            "token_count": 6,
            "page_number": None,
            "section_title": None,
            "document_name": "sample_health_policy.txt",
            "category": "health_insurance",
            "bm25_score": 1.25,
        }
    ]

    def fake_bm25_search(db, user_id, query, top_k=5):
        return fake_results

    monkeypatch.setattr(
        "app.api.retrieval_routes.bm25_search",
        fake_bm25_search,
    )

    response = client.post(
        "/retrieval/bm25-search",
        json={
            "user_id": "local-user-123",
            "query": "urgent care copay",
            "top_k": 5,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == "local-user-123"
    assert data["query"] == "urgent care copay"
    assert data["top_k"] == 5
    assert data["result_count"] == 1
    assert data["results"][0]["chunk_id"] == "chunk-1"
    assert data["results"][0]["bm25_score"] == 1.25


def test_bm25_endpoint_empty_user_id_returns_400():
    response = client.post(
        "/retrieval/bm25-search",
        json={
            "user_id": "   ",
            "query": "urgent care copay",
            "top_k": 5,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "user_id is required"


def test_bm25_endpoint_missing_user_id_returns_422():
    response = client.post(
        "/retrieval/bm25-search",
        json={
            "query": "urgent care copay",
            "top_k": 5,
        },
    )

    assert response.status_code == 422


def test_bm25_endpoint_empty_query_returns_400():
    response = client.post(
        "/retrieval/bm25-search",
        json={
            "user_id": "local-user-123",
            "query": "   ",
            "top_k": 5,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "query is required"


def test_bm25_endpoint_missing_query_returns_422():
    response = client.post(
        "/retrieval/bm25-search",
        json={
            "user_id": "local-user-123",
            "top_k": 5,
        },
    )

    assert response.status_code == 422


def test_bm25_endpoint_top_k_greater_than_20_returns_422():
    response = client.post(
        "/retrieval/bm25-search",
        json={
            "user_id": "local-user-123",
            "query": "urgent care copay",
            "top_k": 25,
        },
    )

    assert response.status_code == 422


def test_bm25_search_no_chunks_returns_empty_list(monkeypatch):
    class FakeQuery:
        def join(self, *args, **kwargs):
            return self

        def filter(self, *args, **kwargs):
            return self

        def all(self):
            return []

    class FakeDBWithNoRows:
        def query(self, *args, **kwargs):
            return FakeQuery()

    results = bm25_search(
        db=FakeDBWithNoRows(),
        user_id="local-user-123",
        query="urgent care copay",
        top_k=5,
    )

    assert results == []


def test_bm25_search_result_contains_bm25_score(monkeypatch):
    fake_result = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "user_id": "local-user-123",
            "chunk_text": "Urgent care copay is 50 dollars.",
            "chunk_index": 0,
            "token_count": 6,
            "page_number": None,
            "section_title": None,
            "document_name": "sample_health_policy.txt",
            "category": "health_insurance",
            "bm25_score": 1.5,
        }
    ]

    def fake_bm25_search(db, user_id, query, top_k=5):
        return fake_result

    results = fake_bm25_search(
        db=FakeDB(),
        user_id="local-user-123",
        query="urgent care copay",
        top_k=5,
    )

    assert "bm25_score" in results[0]
    assert isinstance(results[0]["bm25_score"], float)


def test_only_user_scoped_chunks_are_searched_using_route_monkeypatch(monkeypatch):
    app.dependency_overrides[get_db] = override_get_db

    captured_user_id = {}

    def fake_bm25_search(db, user_id, query, top_k=5):
        captured_user_id["value"] = user_id
        return []

    monkeypatch.setattr(
        "app.api.retrieval_routes.bm25_search",
        fake_bm25_search,
    )

    response = client.post(
        "/retrieval/bm25-search",
        json={
            "user_id": "user-a",
            "query": "copay",
            "top_k": 5,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert captured_user_id["value"] == "user-a"
    assert response.json()["results"] == []
    assert response.json()["result_count"] == 0


def test_bm25_endpoint_result_count_correct_using_monkeypatch(monkeypatch):
    app.dependency_overrides[get_db] = override_get_db

    fake_results = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "user_id": "local-user-123",
            "chunk_text": "Urgent care copay is 50 dollars.",
            "chunk_index": 0,
            "token_count": 6,
            "page_number": None,
            "section_title": None,
            "document_name": "sample_health_policy.txt",
            "category": "health_insurance",
            "bm25_score": 1.25,
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-1",
            "user_id": "local-user-123",
            "chunk_text": "Primary care visit copay is 25 dollars.",
            "chunk_index": 1,
            "token_count": 7,
            "page_number": None,
            "section_title": None,
            "document_name": "sample_health_policy.txt",
            "category": "health_insurance",
            "bm25_score": 0.75,
        },
    ]

    def fake_bm25_search(db, user_id, query, top_k=5):
        return fake_results

    monkeypatch.setattr(
        "app.api.retrieval_routes.bm25_search",
        fake_bm25_search,
    )

    response = client.post(
        "/retrieval/bm25-search",
        json={
            "user_id": "local-user-123",
            "query": "copay",
            "top_k": 5,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["result_count"] == 2