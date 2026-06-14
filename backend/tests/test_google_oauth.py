from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import auth_routes
from app.api.auth_routes import router as auth_router
from app.auth import oauth_google
from app.config.settings import settings
from app.db.database import get_db
from app.models.user import User


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    User.__table__.create(bind=engine, checkfirst=True)
    db = testing_session_local()

    try:
        yield db
    finally:
        db.close()
        User.__table__.drop(bind=engine, checkfirst=True)


@pytest.fixture()
def client(db_session):
    app = FastAPI()
    app.include_router(auth_router)

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


@pytest.fixture()
def configured_google(monkeypatch):
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "google-client-id")
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_SECRET", "google-client-secret")
    monkeypatch.setattr(
        settings,
        "GOOGLE_REDIRECT_URI",
        "http://localhost:8000/auth/google/callback",
    )


def _get_state(client: TestClient) -> str:
    response = client.get("/auth/google/login")
    assert response.status_code == 200
    authorization_url = response.json()["authorization_url"]
    return parse_qs(urlparse(authorization_url).query)["state"][0]


def test_google_login_returns_503_when_not_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", None)
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_SECRET", None)
    monkeypatch.setattr(settings, "GOOGLE_REDIRECT_URI", None)

    response = client.get("/auth/google/login")

    assert response.status_code == 503
    assert "Google OAuth is not configured" in response.json()["detail"]


def test_google_login_returns_authorization_url(client, configured_google):
    response = client.get("/auth/google/login")

    assert response.status_code == 200
    authorization_url = response.json()["authorization_url"]
    query = parse_qs(urlparse(authorization_url).query)

    assert authorization_url.startswith(oauth_google.GOOGLE_AUTHORIZATION_ENDPOINT)
    assert query["client_id"] == ["google-client-id"]
    assert query["redirect_uri"] == [
        "http://localhost:8000/auth/google/callback"
    ]
    assert query["state"]
    assert oauth_google.verify_oauth_state(query["state"][0]) is True
    assert query["prompt"] == ["select_account"]


def test_create_oauth_state_returns_signed_valid_state():
    state = oauth_google.create_oauth_state()

    assert isinstance(state, str)
    assert oauth_google.verify_oauth_state(state) is True


def test_verify_oauth_state_rejects_invalid_state():
    assert oauth_google.verify_oauth_state("invalid-state") is False


def test_verify_oauth_state_rejects_expired_state():
    expired_state = jwt.encode(
        {
            "purpose": "google_oauth",
            "nonce": "expired-nonce",
            "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    assert oauth_google.verify_oauth_state(expired_state) is False


def test_google_callback_returns_400_when_code_missing(client, configured_google):
    state = _get_state(client)

    response = client.get("/auth/google/callback", params={"state": state})

    assert response.status_code == 400
    assert response.json()["detail"] == "Missing Google authorization code"


def test_google_callback_returns_400_when_state_missing(client, configured_google):
    response = client.get("/auth/google/callback", params={"code": "google-code"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Missing Google OAuth state"


def test_google_callback_rejects_invalid_state(client, configured_google):
    response = client.get(
        "/auth/google/callback",
        params={"code": "google-code", "state": "invalid-state"},
    )

    assert response.status_code == 400
    assert "did not match this browser session" in response.json()["detail"]


def test_google_callback_sets_secure_session_and_does_not_store_google_token(
    client,
    configured_google,
    monkeypatch,
):
    captured_user_args = {}
    oauth_user = User(
        id="google-app-user",
        email="oauth@example.com",
        full_name="OAuth User",
        hashed_password=None,
        auth_provider="google",
        provider_user_id="google-user-123",
        is_active=True,
    )

    async def fake_exchange(code):
        assert code == "google-code"
        return {"access_token": "google-access-token", "refresh_token": "do-not-store"}

    async def fake_userinfo(access_token):
        assert access_token == "google-access-token"
        return {
            "id": "google-user-123",
            "email": "OAuth@Example.com",
            "verified_email": True,
            "name": "OAuth User",
        }

    def fake_get_or_create_oauth_user(**kwargs):
        captured_user_args.update(kwargs)
        return oauth_user

    monkeypatch.setattr(auth_routes, "exchange_google_code_for_token", fake_exchange)
    monkeypatch.setattr(auth_routes, "fetch_google_userinfo", fake_userinfo)
    monkeypatch.setattr(
        auth_routes,
        "get_or_create_oauth_user",
        fake_get_or_create_oauth_user,
    )

    response = client.get(
        "/auth/google/callback",
        params={"code": "google-code", "state": _get_state(client)},
        follow_redirects=False,
    )

    assert response.status_code == 307
    assert response.headers["location"] == settings.FRONTEND_URL
    assert settings.AUTH_COOKIE_NAME in response.cookies
    assert "httponly" in response.headers["set-cookie"].lower()
    assert "google-access-token" not in response.text
    assert "refresh_token" not in captured_user_args
    assert "access_token" not in captured_user_args
    assert captured_user_args["provider"] == "google"
    assert captured_user_args["email"] == "oauth@example.com"


def test_validate_google_userinfo_rejects_missing_email():
    with pytest.raises(ValueError, match="missing the email"):
        oauth_google.validate_google_userinfo(
            {
                "id": "google-user-123",
                "verified_email": True,
            }
        )


def test_validate_google_userinfo_rejects_unverified_email():
    with pytest.raises(ValueError, match="not verified"):
        oauth_google.validate_google_userinfo(
            {
                "id": "google-user-123",
                "email": "oauth@example.com",
                "verified_email": False,
            }
        )


def test_validate_google_userinfo_returns_normalized_identity():
    identity = oauth_google.validate_google_userinfo(
        {
            "sub": "google-user-123",
            "email": " OAuth@Example.com ",
            "verified_email": True,
            "name": "OAuth User",
        }
    )

    assert identity == {
        "provider_user_id": "google-user-123",
        "email": "oauth@example.com",
        "full_name": "OAuth User",
    }


def test_google_callback_issues_app_jwt_accepted_by_auth_me(
    client,
    configured_google,
    monkeypatch,
):
    async def fake_exchange(code):
        return {"access_token": "google-access-token"}

    async def fake_userinfo(access_token):
        return {
            "id": "google-user-123",
            "email": "oauth@example.com",
            "verified_email": True,
            "name": "OAuth User",
        }

    monkeypatch.setattr(auth_routes, "exchange_google_code_for_token", fake_exchange)
    monkeypatch.setattr(auth_routes, "fetch_google_userinfo", fake_userinfo)

    callback_response = client.get(
        "/auth/google/callback",
        params={"code": "google-code", "state": _get_state(client)},
        follow_redirects=False,
    )
    me_response = client.get("/auth/me")

    assert callback_response.status_code == 307
    assert me_response.status_code == 200
    assert me_response.json()["email"] == "oauth@example.com"
    assert me_response.json()["auth_provider"] == "google"
