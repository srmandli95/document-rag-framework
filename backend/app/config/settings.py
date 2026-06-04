from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "personal-policy-rag-assistant"
    ENV: str = "local"

    STORAGE_PROVIDER: str = "local"

    DATABASE_URL: str

    JWT_SECRET_KEY: str = "change_me"

    RAW_DOCUMENTS_DIR: str = "/app/data/raw_documents"
    REDACTED_DOCUMENTS_DIR: str = "/app/data/redacted_documents"
    EXTRACTED_TEXT_DIR: str = "/app/data/extracted_text"
    PROCESSED_CHUNKS_DIR: str = "/app/data/processed_chunks"

    EMBEDDING_PROVIDER: str = "local"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    RERANKER_PROVIDER: str = "local"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    RERANKER_TOP_K: int = 8

    LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    ANSWER_TOP_K: int = 5

    MAX_UPLOAD_SIZE_MB: int = 250

    LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
