from pydantic import BaseModel, ConfigDict


class AuthUserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    full_name: str | None = None
    auth_provider: str
    is_active: bool
