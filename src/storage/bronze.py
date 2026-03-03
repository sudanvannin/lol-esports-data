"""
Bronze layer storage implementation.

Bronze layer stores raw, immutable data as received from sources.
Data is partitioned by date and stored in both local filesystem and S3 for redundancy.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.config import get_settings

from .base import (
    LocalStorageBackend,
    S3StorageBackend,
    StorageBackend,
    generate_partition_path,
)

logger = logging.getLogger(__name__)


class BronzeStorage:
    """
    Storage manager for Bronze layer data.

    Implements write-through caching: data is written to both local storage
    and S3 for redundancy. Reads prioritize S3 but fall back to local.
    """

    def __init__(
        self,
        local_backend: StorageBackend | None = None,
        s3_backend: StorageBackend | None = None,
        enable_local: bool = True,
        enable_s3: bool = True,
    ):
        """
        Initialize Bronze storage.

        Args:
            local_backend: Local storage backend (auto-created if None)
            s3_backend: S3 storage backend (auto-created if None)
            enable_local: Whether to write to local storage
            enable_s3: Whether to write to S3
        """
        settings = get_settings()

        self.enable_local = enable_local and settings.storage.local_backup_enabled
        self.enable_s3 = enable_s3

        if self.enable_local and local_backend is None:
            local_backend = LocalStorageBackend(
                base_dir=Path(settings.storage.local_backup_dir) / "bronze"
            )
        self._local = local_backend

        if self.enable_s3 and s3_backend is None:
            try:
                s3_backend = S3StorageBackend(
                    bucket=settings.storage.bronze_bucket,
                    endpoint_url=settings.minio.endpoint,
                    access_key=settings.minio.access_key,
                    secret_key=settings.minio.secret_key,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize S3 backend: {e}")
                self.enable_s3 = False
        self._s3 = s3_backend

    def save(
        self,
        data: dict | list | BaseModel,
        data_type: str,
        identifier: str,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, str | None]:
        """
        Save data to Bronze layer.

        Args:
            data: Data to save (dict, list, or Pydantic model)
            data_type: Category of data (leagues, tournaments, matches, games, live_stats)
            identifier: Unique identifier for this data
            timestamp: Timestamp for partitioning (defaults to now)
            metadata: Optional metadata to include

        Returns:
            Dict with paths where data was saved:
            {"local": "/path/to/file", "s3": "s3://bucket/key"}
        """
        timestamp = timestamp or datetime.utcnow()
        path = generate_partition_path(data_type, identifier, timestamp)

        envelope = self._create_envelope(data, data_type, identifier, timestamp, metadata)
        content = json.dumps(envelope, default=str, indent=2).encode("utf-8")

        result: dict[str, str | None] = {"local": None, "s3": None}

        if self.enable_local and self._local:
            try:
                result["local"] = self._local.write(content, path)
                logger.debug(f"Saved to local: {result['local']}")
            except Exception as e:
                logger.error(f"Failed to save to local storage: {e}")

        if self.enable_s3 and self._s3:
            try:
                result["s3"] = self._s3.write(content, path)
                logger.debug(f"Saved to S3: {result['s3']}")
            except Exception as e:
                logger.warning(f"Failed to save to S3: {e}")

        if not any(result.values()):
            raise RuntimeError(f"Failed to save data to any storage backend: {path}")

        return result

    def save_batch(
        self,
        items: list[tuple[dict | list | BaseModel, str, str]],
        timestamp: datetime | None = None,
    ) -> list[dict[str, str | None]]:
        """
        Save multiple items in batch.

        Args:
            items: List of (data, data_type, identifier) tuples
            timestamp: Common timestamp for all items

        Returns:
            List of result dicts for each item
        """
        timestamp = timestamp or datetime.utcnow()
        return [
            self.save(data, data_type, identifier, timestamp)
            for data, data_type, identifier in items
        ]

    def read(
        self,
        data_type: str,
        identifier: str,
        timestamp: datetime,
        prefer_s3: bool = True,
    ) -> dict | None:
        """
        Read data from Bronze layer.

        Args:
            data_type: Category of data
            identifier: Unique identifier
            timestamp: Timestamp for partitioning
            prefer_s3: Whether to try S3 first

        Returns:
            Parsed data dict or None if not found
        """
        path = generate_partition_path(data_type, identifier, timestamp)

        backends = []
        if prefer_s3 and self.enable_s3 and self._s3:
            backends.append(("s3", self._s3))
        if self.enable_local and self._local:
            backends.append(("local", self._local))
        if not prefer_s3 and self.enable_s3 and self._s3:
            backends.append(("s3", self._s3))

        for name, backend in backends:
            try:
                content = backend.read(path)
                envelope = json.loads(content)
                return envelope.get("data")
            except Exception as e:
                logger.debug(f"Failed to read from {name}: {e}")
                continue

        return None

    def list_data(
        self,
        data_type: str,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> list[str]:
        """
        List available data files.

        Args:
            data_type: Category of data
            year: Filter by year
            month: Filter by month
            day: Filter by day

        Returns:
            List of file paths/keys
        """
        prefix = data_type
        if year:
            prefix += f"/year={year}"
            if month:
                prefix += f"/month={month:02d}"
                if day:
                    prefix += f"/day={day:02d}"

        files = []

        if self.enable_s3 and self._s3:
            try:
                files = self._s3.list_files(prefix)
            except Exception as e:
                logger.warning(f"Failed to list S3 files: {e}")

        if not files and self.enable_local and self._local:
            try:
                files = self._local.list_files(prefix)
            except Exception as e:
                logger.warning(f"Failed to list local files: {e}")

        return files

    def _create_envelope(
        self,
        data: dict | list | BaseModel,
        data_type: str,
        identifier: str,
        timestamp: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a data envelope with metadata.

        The envelope wraps raw data with processing metadata for
        lineage tracking and debugging.
        """
        if isinstance(data, BaseModel):
            data_dict = data.model_dump(mode="json")
        else:
            data_dict = data

        return {
            "_metadata": {
                "data_type": data_type,
                "identifier": identifier,
                "ingested_at": timestamp.isoformat(),
                "source": "lol-esports-pipeline",
                "version": "1.0.0",
                **(metadata or {}),
            },
            "data": data_dict,
        }


class BronzeReader:
    """
    Read-only interface for Bronze layer.

    Useful for downstream processing that shouldn't write to Bronze.
    """

    def __init__(self, storage: BronzeStorage):
        self._storage = storage

    def read(
        self,
        data_type: str,
        identifier: str,
        timestamp: datetime,
    ) -> dict | None:
        """Read data from Bronze layer."""
        return self._storage.read(data_type, identifier, timestamp)

    def list_data(
        self,
        data_type: str,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> list[str]:
        """List available data files."""
        return self._storage.list_data(data_type, year, month, day)

    def iterate_data(
        self,
        data_type: str,
        year: int | None = None,
        month: int | None = None,
    ):
        """
        Iterate over all data files for a given type and time range.

        Yields:
            Tuple of (path, data_dict)
        """
        files = self.list_data(data_type, year, month)

        for path in files:
            # Parse timestamp from path
            # Expected format: type/year=YYYY/month=MM/day=DD/id.json
            parts = path.split("/")
            try:
                year_part = int(parts[1].split("=")[1])
                month_part = int(parts[2].split("=")[1])
                day_part = int(parts[3].split("=")[1])
                identifier = parts[4].replace(".json", "")

                timestamp = datetime(year_part, month_part, day_part)
                data = self.read(data_type, identifier, timestamp)

                if data:
                    yield path, data
            except (IndexError, ValueError) as e:
                logger.warning(f"Failed to parse path {path}: {e}")
                continue
