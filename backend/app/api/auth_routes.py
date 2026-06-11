from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.auth.jwt_handler import create_access_token
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
)


router = APIRouter(prefix="/auth", tags=["Auth"])


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