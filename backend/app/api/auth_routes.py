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
from app.schemas.auth_schema import (
    AuthTokenResponse,
    AuthUserResponse,
    LoginRequest,
    OrganizationCreateRequest,
    OrganizationListResponse,
    OrganizationMemberAddRequest,
    OrganizationResponse,
    OrganizationSelectRequest,
    RegisterRequest,
)
from app.services.user_service import (
    authenticate_local_user,
    create_local_user,
    get_user_by_email,
    get_or_create_oauth_user,
)
from app.services.organization_service import (
    add_organization_member,
    create_organization,
    get_membership,
    list_user_organizations,
)
from app.config.settings import settings


router = APIRouter(prefix="/auth", tags=["Auth"])


def _build_token_response(
    user: User,
    organization_id: str | None = None,
) -> AuthTokenResponse:
    token_data = {
        "sub": user.id,
        "email": user.email,
        "auth_provider": user.auth_provider,
    }
    if organization_id:
        token_data["org_id"] = organization_id
    access_token = create_access_token(
        data=token_data
    )
    return AuthTokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=AuthUserResponse.model_validate(user),
    )


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


@router.post("/register", response_model=AuthTokenResponse)
def register(
    request: RegisterRequest,
    response: Response,
    db: Session = Depends(get_db),
) -> AuthTokenResponse:
    if not settings.ALLOW_LOCAL_REGISTRATION:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Local registration is disabled",
        )
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

    auth_response = _build_token_response(user)
    _set_session_cookie(response, auth_response.access_token)
    return auth_response


@router.post(
    "/organizations/{organization_id}/members",
    response_model=OrganizationResponse,
)
def add_member(
    organization_id: str,
    request: OrganizationMemberAddRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> OrganizationResponse:
    actor_membership = get_membership(
        db,
        user_id=str(current_user.id),
        organization_id=organization_id,
    )
    if actor_membership is None or actor_membership.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Organization admin access is required")
    invited_user = get_user_by_email(db, request.email)
    if invited_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User must sign in before being added")
    membership = add_organization_member(
        db,
        organization_id=organization_id,
        user_id=str(invited_user.id),
        role=request.role,
    )
    organization = next(
        org for org, _ in list_user_organizations(db, user_id=str(invited_user.id))
        if org.id == organization_id
    )
    return OrganizationResponse(id=organization.id, name=organization.name, role=membership.role)


@router.post("/login", response_model=AuthTokenResponse)
def login(
    request: LoginRequest,
    response: Response,
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

    auth_response = _build_token_response(user)
    _set_session_cookie(response, auth_response.access_token)
    return auth_response


@router.get("/me", response_model=AuthUserResponse)
def me(
    current_user: User = Depends(get_current_user),
) -> AuthUserResponse:
    return AuthUserResponse.model_validate(current_user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response) -> None:
    response.delete_cookie(settings.AUTH_COOKIE_NAME, path="/")


@router.get("/organizations", response_model=OrganizationListResponse)
def organizations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> OrganizationListResponse:
    rows = list_user_organizations(db, user_id=str(current_user.id))
    return OrganizationListResponse(
        organizations=[
            OrganizationResponse(id=org.id, name=org.name, role=membership.role)
            for org, membership in rows
        ],
        active_organization_id=getattr(current_user, "active_organization_id", None),
    )


@router.post("/organizations", response_model=OrganizationResponse)
def add_organization(
    request: OrganizationCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> OrganizationResponse:
    organization = create_organization(
        db,
        name=request.name,
        owner_user_id=str(current_user.id),
    )
    return OrganizationResponse(id=organization.id, name=organization.name, role="admin")


@router.post("/organizations/select", response_model=AuthTokenResponse)
def select_organization(
    request: OrganizationSelectRequest,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AuthTokenResponse:
    if request.organization_id is None:
        auth_response = _build_token_response(current_user)
        _set_session_cookie(response, auth_response.access_token)
        return auth_response

    membership = get_membership(
        db,
        user_id=str(current_user.id),
        organization_id=request.organization_id,
    )
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    auth_response = _build_token_response(current_user, request.organization_id)
    _set_session_cookie(response, auth_response.access_token)
    return auth_response


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

    auth_response = _build_token_response(user)
    response = RedirectResponse(settings.FRONTEND_URL)
    _set_session_cookie(response, auth_response.access_token)
    response.delete_cookie(
        settings.OAUTH_STATE_COOKIE_NAME,
        path="/auth/google/callback",
    )
    return response
