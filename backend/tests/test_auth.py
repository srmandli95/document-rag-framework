from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.auth_routes import router as auth_router
from app.auth.jwt_handler import create_access_token, decode_access_token
from app.auth.password_utils import hash_password, verify_password
from app.db.database import get_db
from app.models.user import User
from app.services.user_service import (
    authenticate_local_user,
    create_local_user,
    get_or_create_oauth_user,
)


@pytest.fixture()
def db_session():
    """
    Auth tests only need the users table.

    Do NOT call Base.metadata.create_all() here because that tries to create
    every project table in SQLite, including PostgreSQL-specific columns from
    chat/document models.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )

    User.__table__.create(bind=engine, checkfirst=True)

    db = TestingSessionLocal()

    try:
        yield db
    finally:
        db.close()
        User.__table__.drop(bind=engine, checkfirst=True)


@pytest.fixture()
def auth_test_client(db_session):
    app = FastAPI()
    app.include_router(auth_router)

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    return TestClient(app)


def test_hash_password_returns_non_plain_password():
    hashed = hash_password("password123")

    assert hashed != "password123"
    assert isinstance(hashed, str)
    assert len(hashed) > 0


def test_verify_password_works_for_correct_password():
    hashed = hash_password("password123")

    assert verify_password("password123", hashed) is True


def test_verify_password_fails_for_wrong_password():
    hashed = hash_password("password123")

    assert verify_password("wrong-password", hashed) is False


def test_verify_password_returns_false_if_hashed_password_is_none():
    assert verify_password("password123", None) is False


def test_create_access_token_returns_token():
    token = create_access_token(
        data={
            "sub": "user-123",
            "email": "test@example.com",
            "auth_provider": "local",
        },
        expires_delta=timedelta(minutes=30),
    )

    assert isinstance(token, str)
    assert len(token) > 0


def test_decode_access_token_returns_payload_with_expected_fields():
    token = create_access_token(
        data={
            "sub": "user-123",
            "email": "test@example.com",
            "auth_provider": "local",
        },
        expires_delta=timedelta(minutes=30),
    )

    payload = decode_access_token(token)

    assert payload["sub"] == "user-123"
    assert payload["email"] == "test@example.com"
    assert payload["auth_provider"] == "local"


def test_create_local_user_hashes_password_and_sets_auth_provider_local(db_session):
    user = create_local_user(
        db=db_session,
        email="Test@Example.com",
        password="password123",
        full_name="Test User",
    )

    assert user.id is not None
    assert user.email == "test@example.com"
    assert user.full_name == "Test User"
    assert user.hashed_password != "password123"
    assert user.auth_provider == "local"
    assert user.provider_user_id is None
    assert user.is_active is True


def test_authenticate_local_user_returns_user_for_correct_password(db_session):
    created_user = create_local_user(
        db=db_session,
        email="test@example.com",
        password="password123",
        full_name="Test User",
    )

    authenticated_user = authenticate_local_user(
        db=db_session,
        email="test@example.com",
        password="password123",
    )

    assert authenticated_user is not None
    assert authenticated_user.id == created_user.id


def test_authenticate_local_user_returns_none_for_wrong_password(db_session):
    create_local_user(
        db=db_session,
        email="test@example.com",
        password="password123",
        full_name="Test User",
    )

    authenticated_user = authenticate_local_user(
        db=db_session,
        email="test@example.com",
        password="wrong-password",
    )

    assert authenticated_user is None


def test_get_or_create_oauth_user_creates_oauth_ready_user(db_session):
    user = get_or_create_oauth_user(
        db=db_session,
        provider="google",
        provider_user_id="google-user-123",
        email="oauth@example.com",
        full_name="OAuth User",
    )

    assert user.id is not None
    assert user.email == "oauth@example.com"
    assert user.full_name == "OAuth User"
    assert user.hashed_password is None
    assert user.auth_provider == "google"
    assert user.provider_user_id == "google-user-123"


def test_get_or_create_oauth_user_returns_existing_user_if_email_already_exists(db_session):
    local_user = create_local_user(
        db=db_session,
        email="same@example.com",
        password="password123",
        full_name="Local User",
    )

    oauth_user = get_or_create_oauth_user(
        db=db_session,
        provider="google",
        provider_user_id="google-user-123",
        email="same@example.com",
        full_name="OAuth User",
    )

    assert oauth_user.id == local_user.id
    assert oauth_user.email == "same@example.com"
    assert oauth_user.auth_provider == "local"


def test_auth_register_returns_token_and_user(auth_test_client):
    response = auth_test_client.post(
        "/auth/register",
        json={
            "email": "test@example.com",
            "password": "password123",
            "full_name": "Test User",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["access_token"]
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "test@example.com"
    assert data["user"]["full_name"] == "Test User"
    assert data["user"]["auth_provider"] == "local"
    assert data["user"]["is_active"] is True


def test_duplicate_auth_register_returns_409(auth_test_client):
    payload = {
        "email": "test@example.com",
        "password": "password123",
        "full_name": "Test User",
    }

    first_response = auth_test_client.post("/auth/register", json=payload)
    second_response = auth_test_client.post("/auth/register", json=payload)

    assert first_response.status_code == 200
    assert second_response.status_code == 409


def test_auth_login_returns_token_for_valid_credentials(auth_test_client):
    auth_test_client.post(
        "/auth/register",
        json={
            "email": "test@example.com",
            "password": "password123",
            "full_name": "Test User",
        },
    )

    response = auth_test_client.post(
        "/auth/login",
        json={
            "email": "test@example.com",
            "password": "password123",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["access_token"]
    assert data["user"]["email"] == "test@example.com"


def test_auth_login_returns_401_for_bad_password(auth_test_client):
    auth_test_client.post(
        "/auth/register",
        json={
            "email": "test@example.com",
            "password": "password123",
            "full_name": "Test User",
        },
    )

    response = auth_test_client.post(
        "/auth/login",
        json={
            "email": "test@example.com",
            "password": "wrong-password",
        },
    )

    assert response.status_code == 401


def test_auth_me_returns_current_user_with_valid_token(auth_test_client):
    register_response = auth_test_client.post(
        "/auth/register",
        json={
            "email": "test@example.com",
            "password": "password123",
            "full_name": "Test User",
        },
    )

    assert register_response.status_code == 200

    access_token = register_response.json()["access_token"]

    response = auth_test_client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["email"] == "test@example.com"
    assert data["auth_provider"] == "local"
    assert data["is_active"] is True


def test_auth_me_returns_401_or_403_without_token(auth_test_client):
    response = auth_test_client.get("/auth/me")

    assert response.status_code in [401, 403]