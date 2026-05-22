from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "personal-policy-rag-assistant"
    ENV: str = "local"

    DATABASE_URL: str

    JWT_SECRET_KEY: str = "change_me"

    UPLOAD_DIR: str = "/app/data/uploaded_files"
    MAX_UPLOAD_SIZE_MB: int = 25

    LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
