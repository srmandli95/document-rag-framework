from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.auth.jwt_handler import create_access_token
from app.auth.oauth_google import (
    GoogleOAuthError,
    build_google_authorization_url,
    create_oauth_state,
    exchange_google_code_for_token,
    fetch_google_userinfo,
    get_google_oauth_configured,
    validate_google_userinfo,
    verify_oauth_state,
)
from app.db.database import get_db
from app.models.user import User
from app.schemas.auth_schema import AuthUserResponse
from app.services.user_service import get_or_create_oauth_user
from app.config.settings import settings


router = APIRouter(prefix="/auth", tags=["Auth"])


def _build_access_token(user: User) -> str:
    token_data = {
        "sub": user.id,
        "email": user.email,
        "auth_provider": user.auth_provider,
    }
    return create_access_token(data=token_data)


def _set_session_cookie(response: Response, access_token: str) -> None:
    response.set_cookie(
        key=settings.AUTH_COOKIE_NAME,
        value=access_token,
        httponly=True,
        secure=settings.AUTH_COOKIE_SECURE,
        samesite="lax",
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )


def _validate_google_oauth_state(state_value: str) -> None:
    if not verify_oauth_state(state_value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired Google OAuth state",
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


@router.get("/me", response_model=AuthUserResponse)
def me(
    current_user: User = Depends(get_current_user),
) -> AuthUserResponse:
    return AuthUserResponse.model_validate(current_user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response) -> None:
    response.delete_cookie(settings.AUTH_COOKIE_NAME, path="/")


@router.get("/google/login")
def google_login(response: Response) -> dict[str, str]:
    _require_google_oauth_configured()

    try:
        state_value = create_oauth_state()
        authorization_url = build_google_authorization_url(
            state=state_value
        )
    except GoogleOAuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    response.set_cookie(
        key=settings.OAUTH_STATE_COOKIE_NAME,
        value=state_value,
        httponly=True,
        secure=settings.AUTH_COOKIE_SECURE,
        samesite="lax",
        max_age=600,
        path="/auth/google/callback",
    )
    return {"authorization_url": authorization_url}


@router.get("/google/callback")
async def google_callback(
    code: str | None = None,
    state: str | None = None,
    state_cookie: str | None = Cookie(default=None, alias=settings.OAUTH_STATE_COOKIE_NAME),
    db: Session = Depends(get_db),
) -> RedirectResponse:
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

    if not state_cookie or state != state_cookie:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google OAuth state did not match this browser session",
        )
    _validate_google_oauth_state(state)
    _require_google_oauth_configured()

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

    access_token = _build_access_token(user)
    response = RedirectResponse(settings.FRONTEND_URL)
    _set_session_cookie(response, access_token)
    response.delete_cookie(
        settings.OAUTH_STATE_COOKIE_NAME,
        path="/auth/google/callback",
    )
    return response
