from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.auth.jwt_handler import decode_access_token
from app.config.settings import settings
from app.db.database import get_db
from app.models.user import User


bearer_scheme = HTTPBearer(auto_error=False)


class DevAuthUser:
    """
    Local-only user object used only when DEV_AUTH_DISABLED=True.
    It intentionally exposes the same fields routes usually need.
    """

    def __init__(self, user_id: str):
        self.id = user_id
        self.email = "dev-auth-disabled@example.local"
        self.full_name = "Local Dev User"
        self.is_active = True
        self.auth_provider = "dev"
        self.provider_user_id = user_id


def _credentials_exception(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _get_user_from_token(token: str, db: Session) -> User:
    try:
        payload = decode_access_token(token)
        user = db.query(User).filter(User.id == str(payload["sub"])).first()
    except (KeyError, TypeError, ValueError) as exc:
        raise _credentials_exception(
            "Invalid or expired authentication token"
        ) from exc

    if user is None or not user.is_active:
        raise _credentials_exception("Invalid or expired authentication token")

    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User | DevAuthUser:
    """
    Protected-route dependency.

    Requires:
    Authorization: Bearer <access_token>

    Local-only bypass:
    If DEV_AUTH_DISABLED=True, this returns DEV_AUTH_USER_ID without requiring a token.
    Keep DEV_AUTH_DISABLED=False by default.
    """
    if settings.DEV_AUTH_DISABLED:
        return DevAuthUser(settings.DEV_AUTH_USER_ID)

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _credentials_exception("Authentication required")

    return _get_user_from_token(credentials.credentials, db)


def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User | DevAuthUser | None:
    if settings.DEV_AUTH_DISABLED:
        return DevAuthUser(settings.DEV_AUTH_USER_ID)

    if credentials is None or credentials.scheme.lower() != "bearer":
        return None

    try:
        return _get_user_from_token(credentials.credentials, db)
    except HTTPException:
        return None


def get_current_user_id(
    current_user: User | DevAuthUser = Depends(get_current_user),
) -> str:
    return str(current_user.id)
