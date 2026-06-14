from fastapi import Cookie, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.auth.jwt_handler import decode_access_token
from app.config.settings import settings
from app.db.database import get_db
from app.models.user import User
from app.services.organization_service import get_membership


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

    organization_id = payload.get("org_id")
    if organization_id:
        membership = get_membership(
            db,
            user_id=str(user.id),
            organization_id=str(organization_id),
        )
        if membership is None:
            raise _credentials_exception("Organization access is no longer valid")
        user.active_organization_id = str(organization_id)
        user.organization_role = membership.role

    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session_cookie: str | None = Cookie(default=None, alias=settings.AUTH_COOKIE_NAME),
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

    token = (
        credentials.credentials
        if credentials is not None and credentials.scheme.lower() == "bearer"
        else session_cookie if isinstance(session_cookie, str) else None
    )
    if not token:
        raise _credentials_exception("Authentication required")

    return _get_user_from_token(token, db)


def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session_cookie: str | None = Cookie(default=None, alias=settings.AUTH_COOKIE_NAME),
    db: Session = Depends(get_db),
) -> User | DevAuthUser | None:
    if settings.DEV_AUTH_DISABLED:
        return DevAuthUser(settings.DEV_AUTH_USER_ID)

    token = (
        credentials.credentials
        if credentials is not None and credentials.scheme.lower() == "bearer"
        else session_cookie if isinstance(session_cookie, str) else None
    )
    if not token:
        return None

    try:
        return _get_user_from_token(token, db)
    except HTTPException:
        return None


def get_current_user_id(
    current_user: User | DevAuthUser = Depends(get_current_user),
) -> str:
    return str(current_user.id)


def get_data_scope_id(current_user: User | DevAuthUser) -> str:
    """Return the validated organization scope, or the user's private scope."""
    return str(
        getattr(current_user, "active_organization_id", None)
        or current_user.id
    )


def require_data_scope_editor(current_user: User | DevAuthUser) -> None:
    if (
        getattr(current_user, "active_organization_id", None)
        and getattr(current_user, "organization_role", None) not in {"admin", "editor"}
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization editor access is required",
        )
