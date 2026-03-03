"""Tests for configuration management."""

import os

import pytest

from src.config import (
    APISettings,
    IngestionSettings,
    MinIOSettings,
    Settings,
    StorageSettings,
    get_settings,
)


class TestMinIOSettings:
    """Tests for MinIO settings."""

    def test_default_values(self):
        """Should have sensible defaults."""
        settings = MinIOSettings()
        assert settings.endpoint == "http://localhost:9000"
        assert settings.access_key == "minio"
        assert settings.secret_key == "minio123"
        assert settings.secure is False

    def test_env_override(self, monkeypatch):
        """Should read from environment variables."""
        monkeypatch.setenv("MINIO_ENDPOINT", "http://custom:9000")
        monkeypatch.setenv("MINIO_ACCESS_KEY", "custom_key")

        settings = MinIOSettings()
        assert settings.endpoint == "http://custom:9000"
        assert settings.access_key == "custom_key"


class TestStorageSettings:
    """Tests for storage settings."""

    def test_default_buckets(self):
        """Should have medallion buckets configured."""
        settings = StorageSettings()
        assert settings.bronze_bucket == "bronze"
        assert settings.silver_bucket == "silver"
        assert settings.gold_bucket == "gold"

    def test_local_backup_enabled_by_default(self):
        """Local backup should be enabled by default."""
        settings = StorageSettings()
        assert settings.local_backup_enabled is True


class TestAPISettings:
    """Tests for API settings."""

    def test_default_api_key(self):
        """Should have default API key."""
        settings = APISettings()
        assert settings.lol_esports_api_key is not None
        assert len(settings.lol_esports_api_key) > 0

    def test_timeout_defaults(self):
        """Should have reasonable timeout defaults."""
        settings = APISettings()
        assert settings.request_timeout == 30.0
        assert settings.max_concurrent_requests == 5
        assert settings.max_retries == 3


class TestIngestionSettings:
    """Tests for ingestion settings."""

    def test_default_leagues(self):
        """Should include major leagues by default (using API slugs)."""
        settings = IngestionSettings()
        assert "cblol-brazil" in settings.leagues
        assert "lck" in settings.leagues
        assert "lec" in settings.leagues

    def test_leagues_from_json(self, monkeypatch):
        """Should parse JSON list from env var."""
        monkeypatch.setenv("INGESTION_LEAGUES", '["lck", "lpl", "lec"]')

        settings = IngestionSettings()
        assert settings.leagues == ["lck", "lpl", "lec"]


class TestSettings:
    """Tests for main Settings class."""

    def test_loads_all_subsettings(self):
        """Should aggregate all sub-settings."""
        settings = Settings()

        assert hasattr(settings, "minio")
        assert hasattr(settings, "storage")
        assert hasattr(settings, "api")
        assert hasattr(settings, "spark")
        assert hasattr(settings, "mlflow")
        assert hasattr(settings, "ingestion")

    def test_default_environment(self):
        """Should default to development."""
        settings = Settings()
        assert settings.environment == "development"
        assert settings.debug is False

    def test_get_settings_caching(self):
        """get_settings should return cached instance."""
        # Clear cache first
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2
