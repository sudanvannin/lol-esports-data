"""
Centralized configuration management.

Uses pydantic-settings for type-safe configuration with environment variable support.
All configuration should be accessed through the `get_settings()` function.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MinIOSettings(BaseSettings):
    """MinIO/S3 connection settings."""

    model_config = SettingsConfigDict(env_prefix="MINIO_")

    endpoint: str = Field(default="http://localhost:9000", description="MinIO endpoint URL")
    access_key: str = Field(default="minio", description="Access key")
    secret_key: str = Field(default="minio123", description="Secret key")
    secure: bool = Field(default=False, description="Use HTTPS")


class StorageSettings(BaseSettings):
    """Data lake storage settings."""

    model_config = SettingsConfigDict(env_prefix="STORAGE_")

    bronze_bucket: str = Field(default="bronze", description="Bronze layer bucket")
    silver_bucket: str = Field(default="silver", description="Silver layer bucket")
    gold_bucket: str = Field(default="gold", description="Gold layer bucket")
    local_backup_enabled: bool = Field(default=True, description="Enable local backup")
    local_backup_dir: str = Field(default="data", description="Local backup directory")


class APISettings(BaseSettings):
    """External API settings."""

    model_config = SettingsConfigDict(env_prefix="API_")

    lol_esports_api_key: str = Field(
        default="0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z",
        description="LoL Esports API key",
    )
    request_timeout: float = Field(default=30.0, description="HTTP request timeout")
    max_concurrent_requests: int = Field(default=5, description="Max concurrent requests")
    max_retries: int = Field(default=3, description="Max retry attempts")


class SparkSettings(BaseSettings):
    """Apache Spark settings."""

    model_config = SettingsConfigDict(env_prefix="SPARK_")

    master_url: str = Field(default="local[*]", description="Spark master URL")
    app_name: str = Field(default="lol-esports-data", description="Spark application name")
    executor_memory: str = Field(default="2g", description="Executor memory")
    driver_memory: str = Field(default="1g", description="Driver memory")


class MLflowSettings(BaseSettings):
    """MLflow tracking settings."""

    model_config = SettingsConfigDict(env_prefix="MLFLOW_")

    tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking URI")
    experiment_name: str = Field(default="lol-esports", description="Default experiment name")


class IngestionSettings(BaseSettings):
    """Data ingestion settings."""

    model_config = SettingsConfigDict(env_prefix="INGESTION_")

    leagues: list[str] = Field(
        default=["cblol-brazil", "lck", "lec", "lcs", "lpl", "worlds", "msi"],
        description="League slugs to collect (use API slugs, not display names)",
    )
    oracle_elixir_years: list[str] = Field(
        default=["2024", "2023", "2022"],
        description="Years to download from Oracle's Elixir",
    )
    batch_size: int = Field(default=100, description="Batch size for processing")

    @field_validator("leagues", mode="before")
    @classmethod
    def parse_leagues(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",")]
        return v


class Settings(BaseSettings):
    """Application settings aggregating all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Runtime environment",
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    minio: MinIOSettings = Field(default_factory=MinIOSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    api: APISettings = Field(default_factory=APISettings)
    spark: SparkSettings = Field(default_factory=SparkSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Settings are loaded once and cached for the lifetime of the application.
    Use this function to access configuration throughout the codebase.

    Returns:
        Settings: Application settings instance
    """
    return Settings()
