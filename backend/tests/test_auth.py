from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.auth_routes import router as auth_router
from app.auth.jwt_handler import create_access_token, decode_access_token
from app.config.settings import settings
from app.db.database import get_db
from app.models.user import User
from app.repositories.user_repository import get_or_create_oauth_user


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    User.__table__.create(bind=engine, checkfirst=True)
    db = sessionmaker(bind=engine)()
    try:
        yield db
    finally:
        db.close()
        User.__table__.drop(bind=engine, checkfirst=True)


@pytest.fixture()
def auth_test_client(db_session):
    app = FastAPI()
    app.include_router(auth_router)
    app.dependency_overrides[get_db] = lambda: db_session
    return TestClient(app)


def test_access_token_round_trip():
    token = create_access_token({"sub": "user-123", "auth_provider": "google"})
    payload = decode_access_token(token)
    assert payload["sub"] == "user-123"
    assert payload["auth_provider"] == "google"


def test_decode_access_token_rejects_expired_token():
    token = create_access_token({"sub": "user-123"}, expires_delta=timedelta(minutes=-1))
    with pytest.raises(ValueError, match="Invalid or expired access token"):
        decode_access_token(token)


def test_decode_access_token_rejects_token_missing_sub():
    token = jwt.encode(
        {"email": "test@example.com"},
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    with pytest.raises(ValueError, match="Invalid or expired access token"):
        decode_access_token(token)


def test_get_or_create_google_user(db_session):
    user = get_or_create_oauth_user(
        db=db_session,
        provider="google",
        provider_user_id="google-user-123",
        email="OAuth@Example.com",
        full_name="OAuth User",
    )
    assert user.email == "oauth@example.com"
    assert user.auth_provider == "google"
    assert user.provider_user_id == "google-user-123"


def test_local_password_routes_do_not_exist(auth_test_client):
    assert auth_test_client.post("/auth/login", json={}).status_code == 404
    assert auth_test_client.post("/auth/register", json={}).status_code == 404


def test_auth_me_returns_401_without_session(auth_test_client):
    response = auth_test_client.get("/auth/me")
    assert response.status_code == 401
