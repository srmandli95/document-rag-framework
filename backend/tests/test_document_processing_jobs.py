from datetime import datetime, timedelta
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import document_routes
from app.auth.dependencies import get_current_user
from app.models.document_processing_job import DocumentProcessingJob
from app.services.document_processing_job_service import (
    create_processing_job,
    get_latest_processing_job_for_document,
    get_processing_job_by_id,
    get_processing_jobs_by_document,
    mark_job_completed,
    mark_job_failed,
    mark_job_running,
    mark_job_skipped,
    update_job_step,
)


class FakeDB:
    def __init__(self):
        self.saved = []
        self.query_result = []

    def add(self, obj):
        if isinstance(obj, DocumentProcessingJob) and obj.id is None:
            obj.id = f"job-{len(self.saved) + 1}"
        self.saved.append(obj)

    def commit(self):
        pass

    def refresh(self, obj):
        pass

    def query(self, model):
        return FakeQuery(self.query_result)


class FakeQuery:
    def __init__(self, results):
        self.results = results

    def filter(self, *conditions):
        for condition in conditions:
            field_name = condition.left.name
            expected = condition.right.value
            self.results = [
                item
                for item in self.results
                if getattr(item, field_name) == expected
            ]
        return self

    def order_by(self, *columns):
        self.results = sorted(
            self.results,
            key=lambda job: job.created_at,
            reverse=True,
        )
        return self

    def all(self):
        return self.results

    def first(self):
        return self.results[0] if self.results else None


def make_job(**overrides):
    values = {
        "id": "job-1",
        "document_id": "doc-1",
        "user_id": "user-1",
        "status": "pending",
        "force": False,
        "current_step": None,
        "steps": [],
        "error_message": None,
        "started_at": None,
        "completed_at": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_create_processing_job_creates_pending_job():
    job = create_processing_job(FakeDB(), "doc-1", "user-1", force=True)
    assert job.id == "job-1"
    assert job.status == "pending"
    assert job.force is True
    assert job.steps == []


def test_mark_job_running_sets_status_and_started_at():
    job = make_job()
    mark_job_running(FakeDB(), job, "extract")
    assert job.status == "running"
    assert job.current_step == "extract"
    assert job.started_at is not None


def test_update_job_step_appends_step():
    job = make_job(steps=[{"name": "extract", "status": "completed", "message": "ok"}])
    update_job_step(
        FakeDB(),
        job,
        {"name": "chunk", "status": "completed", "message": "ok"},
        "chunk",
    )
    assert [step["name"] for step in job.steps] == ["extract", "chunk"]


def test_terminal_job_transitions():
    completed = make_job()
    mark_job_completed(FakeDB(), completed, [])
    assert completed.status == "completed"
    assert completed.completed_at is not None

    failed = make_job()
    mark_job_failed(FakeDB(), failed, [], "boom")
    assert failed.status == "failed"
    assert failed.error_message == "boom"

    skipped = make_job()
    mark_job_skipped(FakeDB(), skipped, [], "already done")
    assert skipped.status == "skipped"
    assert skipped.completed_at is not None


def test_processing_job_queries_return_newest_first():
    db = FakeDB()
    older = make_job(id="older", created_at=datetime.utcnow() - timedelta(minutes=1))
    newer = make_job(id="newer", created_at=datetime.utcnow())
    db.query_result = [older, newer]

    assert [job.id for job in get_processing_jobs_by_document(db, "doc-1", "user-1")] == [
        "newer",
        "older",
    ]
    assert get_processing_job_by_id(db, "newer", "user-1").id == "newer"
    assert get_latest_processing_job_for_document(db, "doc-1", "user-1").id == "newer"


def test_processing_job_queries_scope_by_user_id():
    db = FakeDB()
    db.query_result = [
        make_job(id="owner-job", user_id="user-1"),
        make_job(id="other-job", user_id="user-2"),
    ]

    jobs = get_processing_jobs_by_document(db, "doc-1", "user-1")
    assert [job.id for job in jobs] == ["owner-job"]
    assert get_processing_job_by_id(db, "other-job", "user-1") is None


def _client(monkeypatch, document, jobs=None, job=None):
    app = FastAPI()
    app.include_router(document_routes.router)

    def fake_get_db():
        yield FakeDB()

    app.dependency_overrides[document_routes.get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="user-1")
    monkeypatch.setattr(document_routes, "get_document_by_id", lambda **kwargs: document)
    monkeypatch.setattr(
        document_routes,
        "get_processing_jobs_by_document",
        lambda db, document_id, user_id: jobs or [],
    )
    monkeypatch.setattr(
        document_routes,
        "get_processing_job_by_id",
        lambda db, job_id, user_id: job,
    )
    return TestClient(app)


def test_list_processing_jobs_returns_owner_jobs(monkeypatch):
    document = SimpleNamespace(id="doc-1", user_id="user-1", status="uploaded")
    client = _client(monkeypatch, document, jobs=[make_job()])
    response = client.get("/documents/doc-1/processing-jobs")
    assert response.status_code == 200
    assert response.json()["jobs"][0]["job_id"] == "job-1"


def test_list_processing_jobs_returns_404_for_missing_document(monkeypatch):
    response = _client(monkeypatch, None).get("/documents/doc-1/processing-jobs")
    assert response.status_code == 404


def test_get_processing_job_returns_detail(monkeypatch):
    response = _client(monkeypatch, None, job=make_job()).get(
        "/documents/processing-jobs/job-1"
    )
    assert response.status_code == 200
    assert response.json()["job_id"] == "job-1"


def test_get_processing_job_returns_404_for_other_user(monkeypatch):
    response = _client(monkeypatch, None, job=None).get(
        "/documents/processing-jobs/other-job"
    )
    assert response.status_code == 404
