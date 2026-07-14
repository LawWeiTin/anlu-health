from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    app_name: str = "Anlu Health"
    app_base_url: str = "http://localhost:8000"
    database_url: str = "sqlite:///./anlu.db"
    cookie_secure: bool = False
    session_days: int = Field(default=7, ge=1, le=30)
    allow_registration: bool = True
    save_chat_history: bool = False
    data_encryption_key: str | None = None
    emergency_number: str = "995"
    emergency_region: str = "Singapore"

    model_provider: Literal["mock", "openai_compatible"] = "mock"
    model_api_url: str | None = None
    model_api_token: str | None = None
    hf_token: str | None = None
    model_name: str = "google/medgemma-1.5-4b-it"
    model_timeout_seconds: float = Field(default=45, ge=5, le=180)

    embedding_provider: Literal["mock", "openai_compatible", "tei"] = "mock"
    embedding_api_url: str | None = None
    embedding_api_token: str | None = None
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimensions: int = Field(default=384, ge=128, le=4096)

    redis_url: str | None = None
    sentry_dsn: str | None = None
    metrics_token: str | None = None
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_production(self) -> "Settings":
        if self.embedding_dimensions != 384:
            raise ValueError("This schema is pinned to 384-dimensional embeddings")
        if self.save_chat_history and not self.data_encryption_key:
            raise ValueError("DATA_ENCRYPTION_KEY is required when chat history is enabled")
        if self.app_env == "production":
            if self.model_provider == "mock" or self.embedding_provider == "mock":
                raise ValueError("Mock model/embeddings are not allowed in production")
            if not self.model_api_url or not self.embedding_api_url:
                raise ValueError("Production model and embedding API URLs are required")
            if not self.hf_token and not (self.model_api_token and self.embedding_api_token):
                raise ValueError("HF_TOKEN or separate model and embedding tokens are required")
            if not self.cookie_secure:
                raise ValueError("Secure cookies are required in production")
            if not self.metrics_token:
                raise ValueError("METRICS_TOKEN is required in production")
        return self

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url.startswith("postgresql://"):
            return self.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        if self.database_url.startswith("postgres://"):
            return self.database_url.replace("postgres://", "postgresql+psycopg://", 1)
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
