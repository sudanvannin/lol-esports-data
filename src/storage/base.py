"""
Abstract base classes for storage backends.

Follows the Repository pattern to decouple storage implementation from business logic.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class Serializer(Protocol):
    """Protocol for data serializers."""

    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes."""
        ...

    def deserialize(self, data: bytes, model: type[T]) -> T:
        """Deserialize bytes to model."""
        ...


class JSONSerializer:
    """JSON serializer implementation."""

    def serialize(self, data: Any) -> bytes:
        """Serialize data to JSON bytes."""
        if isinstance(data, BaseModel):
            return data.model_dump_json(indent=2).encode("utf-8")
        return json.dumps(data, default=str, indent=2).encode("utf-8")

    def deserialize(self, data: bytes, model: type[T]) -> T:
        """Deserialize JSON bytes to model."""
        return model.model_validate_json(data)


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.

    Implementations should handle the specifics of different storage systems
    (local filesystem, S3, GCS, etc.) while exposing a consistent interface.
    """

    def __init__(self, serializer: Serializer | None = None):
        self.serializer = serializer or JSONSerializer()

    @abstractmethod
    def write(
        self,
        data: bytes,
        path: str,
        content_type: str = "application/json",
    ) -> str:
        """
        Write raw bytes to storage.

        Args:
            data: Raw bytes to write
            path: Destination path/key
            content_type: MIME type of the content

        Returns:
            Full path/URI where data was written
        """
        ...

    @abstractmethod
    def read(self, path: str) -> bytes:
        """
        Read raw bytes from storage.

        Args:
            path: Source path/key

        Returns:
            Raw bytes content
        """
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a path exists in storage."""
        ...

    @abstractmethod
    def list_files(self, prefix: str) -> list[str]:
        """List files matching a prefix."""
        ...

    @abstractmethod
    def delete(self, path: str) -> bool:
        """Delete a file from storage."""
        ...

    def write_model(
        self,
        model: BaseModel,
        path: str,
    ) -> str:
        """
        Write a Pydantic model to storage.

        Args:
            model: Pydantic model instance
            path: Destination path

        Returns:
            Full path where data was written
        """
        data = self.serializer.serialize(model)
        return self.write(data, path, content_type="application/json")

    def read_model(self, path: str, model_class: type[T]) -> T:
        """
        Read and parse a Pydantic model from storage.

        Args:
            path: Source path
            model_class: Pydantic model class to parse into

        Returns:
            Parsed model instance
        """
        data = self.read(path)
        return self.serializer.deserialize(data, model_class)


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(
        self,
        base_dir: str | Path,
        serializer: Serializer | None = None,
    ):
        super().__init__(serializer)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path: str) -> Path:
        """Resolve relative path to absolute."""
        return self.base_dir / path

    def write(
        self,
        data: bytes,
        path: str,
        content_type: str = "application/json",
    ) -> str:
        """Write bytes to local filesystem."""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)
        logger.debug(f"Wrote {len(data)} bytes to {full_path}")
        return str(full_path)

    def read(self, path: str) -> bytes:
        """Read bytes from local filesystem."""
        full_path = self._resolve_path(path)
        return full_path.read_bytes()

    def exists(self, path: str) -> bool:
        """Check if path exists locally."""
        return self._resolve_path(path).exists()

    def list_files(self, prefix: str) -> list[str]:
        """List files matching prefix."""
        base = self._resolve_path(prefix)
        if base.is_file():
            return [str(base.relative_to(self.base_dir))]

        if not base.exists():
            return []

        return [
            str(p.relative_to(self.base_dir))
            for p in base.rglob("*")
            if p.is_file()
        ]

    def delete(self, path: str) -> bool:
        """Delete file from local filesystem."""
        full_path = self._resolve_path(path)
        if full_path.exists():
            full_path.unlink()
            return True
        return False


class S3StorageBackend(StorageBackend):
    """S3-compatible storage backend (works with MinIO, AWS S3, etc.)."""

    def __init__(
        self,
        bucket: str,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        serializer: Serializer | None = None,
    ):
        super().__init__(serializer)
        self.bucket = bucket
        self._client = None
        self._endpoint_url = endpoint_url
        self._access_key = access_key
        self._secret_key = secret_key

    @property
    def client(self):
        """Lazy initialization of S3 client."""
        if self._client is None:
            import boto3
            from botocore.client import Config

            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint_url,
                aws_access_key_id=self._access_key,
                aws_secret_access_key=self._secret_key,
                config=Config(signature_version="s3v4"),
            )
        return self._client

    def write(
        self,
        data: bytes,
        path: str,
        content_type: str = "application/json",
    ) -> str:
        """Write bytes to S3."""
        self.client.put_object(
            Bucket=self.bucket,
            Key=path,
            Body=data,
            ContentType=content_type,
        )
        uri = f"s3://{self.bucket}/{path}"
        logger.debug(f"Wrote {len(data)} bytes to {uri}")
        return uri

    def read(self, path: str) -> bytes:
        """Read bytes from S3."""
        response = self.client.get_object(Bucket=self.bucket, Key=path)
        return response["Body"].read()

    def exists(self, path: str) -> bool:
        """Check if key exists in S3."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=path)
            return True
        except self.client.exceptions.ClientError:
            return False

    def list_files(self, prefix: str) -> list[str]:
        """List objects matching prefix."""
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]

    def delete(self, path: str) -> bool:
        """Delete object from S3."""
        try:
            self.client.delete_object(Bucket=self.bucket, Key=path)
            return True
        except Exception:
            return False


def generate_partition_path(
    data_type: str,
    identifier: str,
    timestamp: datetime | None = None,
    file_extension: str = "json",
) -> str:
    """
    Generate a partitioned storage path.

    Args:
        data_type: Type of data (leagues, matches, games, etc.)
        identifier: Unique identifier for the file
        timestamp: Timestamp for date partitioning
        file_extension: File extension

    Returns:
        Partitioned path string (e.g., "matches/year=2024/month=02/day=17/match_123.json")
    """
    timestamp = timestamp or datetime.utcnow()
    return (
        f"{data_type}/"
        f"year={timestamp.year}/"
        f"month={timestamp.month:02d}/"
        f"day={timestamp.day:02d}/"
        f"{identifier}.{file_extension}"
    )
