from fastapi import Cookie, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.auth.jwt_handler import decode_access_token
from app.config.settings import settings
from app.db.database import get_db
from app.models.user import User
from app.utils.logger import get_logger


bearer_scheme = HTTPBearer(auto_error=False)
logger = get_logger(__name__)


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
        logger.warning("Authentication token rejected during decoding")
        raise _credentials_exception(
            "Invalid or expired authentication token"
        ) from exc

    if user is None or not user.is_active:
        logger.warning("Authentication token rejected: user missing or inactive")
        raise _credentials_exception("Invalid or expired authentication token")

    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session_cookie: str | None = Cookie(default=None, alias=settings.AUTH_COOKIE_NAME),
    db: Session = Depends(get_db),
) -> User:
    """
    Protected-route dependency.

    Requires:
    Authorization: Bearer <access_token>

    Browser requests normally authenticate with the HttpOnly session cookie.
    Bearer tokens remain supported for API clients.
    """
    token = (
        credentials.credentials
        if credentials is not None and credentials.scheme.lower() == "bearer"
        else session_cookie if isinstance(session_cookie, str) else None
    )
    if not token:
        logger.info(
            "Authentication required but no bearer token or session cookie was sent"
        )
        raise _credentials_exception("Authentication required")

    user = _get_user_from_token(token, db)
    logger.info(
        "Authenticated request accepted (user_id=%s, source=%s)",
        user.id,
        "bearer" if credentials is not None else "cookie",
    )
    return user


def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session_cookie: str | None = Cookie(default=None, alias=settings.AUTH_COOKIE_NAME),
    db: Session = Depends(get_db),
) -> User | None:
    token = (
        credentials.credentials
        if credentials is not None and credentials.scheme.lower() == "bearer"
        else session_cookie if isinstance(session_cookie, str) else None
    )
    if not token:
        logger.info(
            "Optional authentication skipped: no bearer token or session cookie"
        )
        return None

    try:
        user = _get_user_from_token(token, db)
        logger.info(
            "Optional authenticated request accepted (user_id=%s, source=%s)",
            user.id,
            "bearer" if credentials is not None else "cookie",
        )
        return user
    except HTTPException:
        logger.info("Optional authentication ignored an invalid token")
        return None


def get_current_user_id(
    current_user: User = Depends(get_current_user),
) -> str:
    return str(current_user.id)
