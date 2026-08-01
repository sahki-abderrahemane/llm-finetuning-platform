"""
Application settings loaded from environment variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ------------------------------------------------------------------
    # Credentials
    # ------------------------------------------------------------------

    HUGGINGFACE_TOKEN: str | None = None
    OPENAI_API_KEY: str | None = None
    MENTORAI_API_KEY: str | None = None

    # ------------------------------------------------------------------
    # Model / Dataset
    # ------------------------------------------------------------------

    MODEL_NAME: str = "Qwen/Qwen2.5-1.5B-Instruct"
    TRAIN_DATASET: Path = Path("data/processed/train.json")
    VALIDATION_DATASET: Path | None = Path("data/processed/validation.json")
    OUTPUT_DIR: Path = Path("models/checkpoints")

    # ------------------------------------------------------------------
    # Deployment
    # ------------------------------------------------------------------

    DEPLOY_BACKEND: str = "vllm"
    DEPLOY_ADAPTER_PATH: Path | None = None
    DEPLOY_MERGED_MODEL_PATH: Path | None = None

    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95

    # ------------------------------------------------------------------
    # Experiment Tracking
    # ------------------------------------------------------------------

    EXPERIMENT_NAME: str | None = None
    EXPERIMENT_TAG: str | None = None

    # ------------------------------------------------------------------
    # Logging / Server
    # ------------------------------------------------------------------

    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
