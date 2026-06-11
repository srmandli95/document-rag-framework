from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

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


def _credentials_exception(detail: str = "Could not validate credentials") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _decode_access_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError as exc:
        raise _credentials_exception("Invalid or expired token") from exc

    return payload


def _extract_user_id_from_payload(payload: dict[str, Any]) -> str:
    user_id = payload.get("sub") or payload.get("user_id")

    if not user_id:
        raise _credentials_exception("Token missing user identity")

    return str(user_id)


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
        raise _credentials_exception("Missing Authorization Bearer token")

    payload = _decode_access_token(credentials.credentials)
    user_id = _extract_user_id_from_payload(payload)

    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        raise _credentials_exception("User not found")

    if hasattr(user, "is_active") and not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def get_current_user_id(
    current_user: User | DevAuthUser = Depends(get_current_user),
) -> str:
    return str(current_user.id)