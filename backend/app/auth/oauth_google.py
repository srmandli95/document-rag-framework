from datetime import datetime, timedelta, timezone
from typing import Any
import uuid

import httpx
from authlib.integrations.httpx_client import AsyncOAuth2Client, OAuth2Client
from jose import JWTError, jwt

from app.config.settings import settings
from app.utils.logger import get_logger


GOOGLE_AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v2/userinfo"
GOOGLE_SCOPES = "openid email profile"
GOOGLE_OAUTH_STATE_EXPIRE_MINUTES = 10
logger = get_logger(__name__)


class GoogleOAuthError(Exception):
    """Raised when Google OAuth configuration or provider requests fail."""


def _mask_client_id(client_id: str | None) -> str:
    if not client_id:
        return "missing"
    if len(client_id) <= 12:
        return "***"
    return f"{client_id[:6]}...{client_id[-6:]}"


def create_oauth_state() -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=GOOGLE_OAUTH_STATE_EXPIRE_MINUTES
    )
    payload = {
        "purpose": "google_oauth",
        "nonce": str(uuid.uuid4()),
        "exp": expires_at,
    }
    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def verify_oauth_state(state: str) -> bool:
    try:
        payload = jwt.decode(
            state,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except (JWTError, TypeError, ValueError):
        logger.warning("Google OAuth state verification failed")
        return False

    is_valid = (
        payload.get("purpose") == "google_oauth"
        and bool(payload.get("nonce"))
    )
    if not is_valid:
        logger.warning("Google OAuth state payload was invalid")
    return is_valid


def get_google_oauth_configured() -> bool:
    return bool(
        settings.GOOGLE_CLIENT_ID
        and settings.GOOGLE_CLIENT_SECRET
        and settings.GOOGLE_REDIRECT_URI
    )


def _require_google_oauth_configured() -> None:
    if not get_google_oauth_configured():
        logger.warning(
            "Google OAuth is not fully configured "
            "(client_id=%s, client_secret=%s, redirect_uri=%s)",
            "present" if settings.GOOGLE_CLIENT_ID else "missing",
            "present" if settings.GOOGLE_CLIENT_SECRET else "missing",
            settings.GOOGLE_REDIRECT_URI or "missing",
        )
        raise GoogleOAuthError(
            "Google OAuth is not configured. Set GOOGLE_CLIENT_ID, "
            "GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI."
        )


def build_google_authorization_url(state: str) -> str:
    _require_google_oauth_configured()
    logger.info(
        "Building Google OAuth authorization URL "
        "(client_id=%s, redirect_uri=%s, scopes=%s)",
        _mask_client_id(settings.GOOGLE_CLIENT_ID),
        settings.GOOGLE_REDIRECT_URI,
        GOOGLE_SCOPES,
    )

    with OAuth2Client(
        client_id=settings.GOOGLE_CLIENT_ID,
        redirect_uri=settings.GOOGLE_REDIRECT_URI,
        scope=GOOGLE_SCOPES,
    ) as client:
        authorization_url, _ = client.create_authorization_url(
            GOOGLE_AUTHORIZATION_ENDPOINT,
            state=state,
            access_type="offline",
            prompt="select_account",
        )
    return authorization_url


async def exchange_google_code_for_token(code: str) -> dict[str, Any]:
    _require_google_oauth_configured()
    logger.info(
        "Exchanging Google authorization code for token "
        "(redirect_uri=%s)",
        settings.GOOGLE_REDIRECT_URI,
    )

    try:
        async with AsyncOAuth2Client(
            client_id=settings.GOOGLE_CLIENT_ID,
            client_secret=settings.GOOGLE_CLIENT_SECRET,
            redirect_uri=settings.GOOGLE_REDIRECT_URI,
        ) as client:
            token = await client.fetch_token(
                GOOGLE_TOKEN_ENDPOINT,
                code=code,
                grant_type="authorization_code",
            )
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Google token exchange failed: %s", exc)
        raise GoogleOAuthError(f"Google token exchange failed: {exc}") from exc
    except Exception as exc:
        logger.exception("Google token exchange failed unexpectedly")
        raise GoogleOAuthError("Google token exchange failed") from exc

    if not token.get("access_token"):
        logger.warning("Google token response was missing an access token")
        raise GoogleOAuthError("Google token response did not include an access token")

    logger.info(
        "Google token exchange succeeded (token_fields=%s)",
        sorted(token.keys()),
    )
    return dict(token)


async def fetch_google_userinfo(access_token: str) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info("Fetching Google userinfo")
            response = await client.get(
                GOOGLE_USERINFO_ENDPOINT,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            response.raise_for_status()
            userinfo = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Google userinfo request failed: %s", exc)
        raise GoogleOAuthError(f"Google user info request failed: {exc}") from exc

    if not isinstance(userinfo, dict):
        logger.warning("Google userinfo response was not an object")
        raise GoogleOAuthError("Google user info response was invalid")

    logger.info(
        "Google userinfo fetched (has_email=%s, verified_email=%s)",
        bool(userinfo.get("email")),
        userinfo.get("verified_email"),
    )
    return userinfo


def validate_google_userinfo(userinfo: dict[str, Any]) -> dict[str, str | None]:
    provider_user_id = userinfo.get("id") or userinfo.get("sub")
    email = userinfo.get("email")

    if not provider_user_id:
        raise ValueError("Google user info is missing the user ID")

    if not email or not str(email).strip():
        raise ValueError("Google user info is missing the email address")

    if userinfo.get("verified_email") is False:
        raise ValueError("Google email address is not verified")

    full_name = userinfo.get("name")

    return {
        "provider_user_id": str(provider_user_id),
        "email": str(email).strip().lower(),
        "full_name": str(full_name).strip() if full_name else None,
    }
