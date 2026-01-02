"""Application settings loaded from environment variables."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration using Pydantic Settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI API
    openai_api_key: str

    # Model names
    gpt4o_model: str = "gpt-4o"
    gpt4o_mini_model: str = "gpt-4o-mini"

    # Cache settings
    cache_ttl_seconds: int = 86400  # 24 hours

    # Agent settings
    max_replan_iterations: int = 3

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
