"""Shared pytest fixtures and utilities for Day 20 authorization tests."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user
from app.db.database import get_db
from app.main import app


@dataclass
class FakeUser:
    """Fake user object compatible with User and DevAuthUser."""
    id: str
    email: str = "test-user@example.com"
    full_name: str = "Test User"
    is_active: bool = True
    auth_provider: str = "local"
    provider_user_id: str | None = None


class FakeDB:
    """Minimal fake database for unit tests."""
    pass


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides():
    """Provide auth compatibility for older route tests and isolate overrides."""
    app.dependency_overrides.clear()
    app.dependency_overrides[get_current_user] = legacy_request_user
    yield
    app.dependency_overrides.clear()


def override_get_db():
    """Override database dependency with fake DB."""
    yield FakeDB()


def override_get_current_user(user_id: str = "test-user"):
    """Factory to create a get_current_user override with specified user_id."""
    def _override():
        return FakeUser(id=user_id)
    return _override


async def legacy_request_user(request: Request):
    """Authenticate older route tests from their now-ignored user_id field."""
    user_id = request.query_params.get("user_id")

    if not user_id:
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            payload = await request.json()
            user_id = payload.get("user_id")
        elif (
            content_type.startswith("multipart/form-data")
            or content_type.startswith("application/x-www-form-urlencoded")
        ):
            form = await request.form()
            user_id = form.get("user_id")

    if not user_id or not str(user_id).strip():
        raise HTTPException(status_code=401, detail="Missing test user identity")

    return FakeUser(id=str(user_id).strip())


def setup_auth_overrides(user_id: str = "test-user"):
    """Setup both DB and auth overrides."""
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user(user_id)


def setup_db_override_only():
    """Setup only database override (for public endpoints)."""
    app.dependency_overrides[get_db] = override_get_db
