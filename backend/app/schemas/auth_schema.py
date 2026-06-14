from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class AuthUserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    full_name: str | None = None
    auth_provider: str
    is_active: bool


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUserResponse


class OrganizationCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class OrganizationSelectRequest(BaseModel):
    organization_id: str | None = None


class OrganizationMemberAddRequest(BaseModel):
    email: EmailStr
    role: str = "member"


class OrganizationResponse(BaseModel):
    id: str
    name: str
    role: str


class OrganizationListResponse(BaseModel):
    organizations: list[OrganizationResponse]
    active_organization_id: str | None = None
