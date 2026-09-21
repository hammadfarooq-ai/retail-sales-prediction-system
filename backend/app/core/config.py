"""Application settings, read from environment variables / .env (never hard-coded secrets)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Retail Sales Prediction API"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", description="development | production | test")
    log_level: str = "INFO"
    log_json: bool = False

    # Database ---------------------------------------------------------------------------
    database_url: str | None = None  # full SQLAlchemy URL; overrides the parts below when set
    postgres_user: str = "retail"
    postgres_password: str = "change-me"
    postgres_db: str = "retail_sales"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    db_pool_size: int = 5
    db_echo: bool = False

    # Model ------------------------------------------------------------------------------
    model_artifacts_dir: Path = PROJECT_ROOT / "ml" / "artifacts"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    history_days_in_db: int = 180  # days of item-store history kept in PostgreSQL
    max_forecast_horizon_days: int = 28
    max_batch_size: int = 500

    # HTTP -------------------------------------------------------------------------------
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @field_validator("model_artifacts_dir", "processed_data_dir")
    @classmethod
    def _resolve_relative(cls, v: Path) -> Path:
        """Relative paths in .env are relative to the project root, not the CWD."""
        return v if v.is_absolute() else (PROJECT_ROOT / v).resolve()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
