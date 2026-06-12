from datetime import timedelta
from html import escape
import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.auth.jwt_handler import create_access_token, decode_access_token
from app.auth.oauth_google import (
    GoogleOAuthError,
    build_google_authorization_url,
    exchange_google_code_for_token,
    fetch_google_userinfo,
    get_google_oauth_configured,
    validate_google_userinfo,
)
from app.db.database import get_db
from app.models.user import User
from app.schemas.auth_schema import (
    AuthTokenResponse,
    AuthUserResponse,
    LoginRequest,
    RegisterRequest,
)
from app.services.user_service import (
    authenticate_local_user,
    create_local_user,
    get_user_by_email,
    get_or_create_oauth_user,
)


router = APIRouter(prefix="/auth", tags=["Auth"])
GOOGLE_OAUTH_STATE_EXPIRE_MINUTES = 10


def _build_token_response(user: User) -> AuthTokenResponse:
    access_token = create_access_token(
        data={
            "sub": user.id,
            "email": user.email,
            "auth_provider": user.auth_provider,
        }
    )

    return AuthTokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=AuthUserResponse.model_validate(user),
    )


def _create_google_oauth_state() -> str:
    return create_access_token(
        data={
            "purpose": "google_oauth_state",
            "nonce": secrets.token_urlsafe(24),
        },
        expires_delta=timedelta(minutes=GOOGLE_OAUTH_STATE_EXPIRE_MINUTES),
    )


def _validate_google_oauth_state(state_value: str) -> None:
    try:
        payload = decode_access_token(state_value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired Google OAuth state",
        ) from exc

    if payload.get("purpose") != "google_oauth_state" or not payload.get("nonce"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Google OAuth state",
        )


def _require_google_oauth_configured() -> None:
    if not get_google_oauth_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Google OAuth is not configured. Set GOOGLE_CLIENT_ID, "
                "GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI."
            ),
        )


@router.post("/register", response_model=AuthTokenResponse)
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
) -> AuthTokenResponse:
    existing_user = get_user_by_email(db, request.email)

    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists",
        )

    try:
        user = create_local_user(
            db=db,
            email=request.email,
            password=request.password,
            full_name=request.full_name,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return _build_token_response(user)


@router.post("/login", response_model=AuthTokenResponse)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
) -> AuthTokenResponse:
    user = authenticate_local_user(
        db=db,
        email=request.email,
        password=request.password,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _build_token_response(user)


@router.get("/me", response_model=AuthUserResponse)
def me(
    current_user: User = Depends(get_current_user),
) -> AuthUserResponse:
    return AuthUserResponse.model_validate(current_user)


@router.get("/google/login")
def google_login() -> dict[str, str]:
    _require_google_oauth_configured()

    try:
        authorization_url = build_google_authorization_url(
            state=_create_google_oauth_state()
        )
    except GoogleOAuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return {"authorization_url": authorization_url}


@router.get("/google/callback", response_class=HTMLResponse)
async def google_callback(
    code: str | None = None,
    state: str | None = None,
    db: Session = Depends(get_db),
) -> HTMLResponse:
    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Google authorization code",
        )

    if not state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Google OAuth state",
        )

    _require_google_oauth_configured()
    _validate_google_oauth_state(state)

    try:
        google_token = await exchange_google_code_for_token(code)
        google_access_token = google_token.get("access_token")
        if not google_access_token:
            raise GoogleOAuthError(
                "Google token response did not include an access token"
            )
        userinfo = await fetch_google_userinfo(str(google_access_token))
    except GoogleOAuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    try:
        identity = validate_google_userinfo(userinfo)
        user = get_or_create_oauth_user(
            db=db,
            provider="google",
            provider_user_id=str(identity["provider_user_id"]),
            email=str(identity["email"]),
            full_name=identity["full_name"],
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    auth_response = _build_token_response(user)
    email = escape(user.email)
    auth_provider = escape(user.auth_provider)
    access_token = escape(auth_response.access_token)

    return HTMLResponse(
        content=f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Google login successful</title></head>
<body>
  <h1>Google login successful</h1>
  <p>Email: <strong>{email}</strong></p>
  <p>Auth provider: <strong>{auth_provider}</strong></p>
  <p>This app access token is for local development only:</p>
  <pre>{access_token}</pre>
  <p>Copy this token and paste it in Chainlit:</p>
  <pre>/token {access_token}</pre>
</body>
</html>"""
    )
