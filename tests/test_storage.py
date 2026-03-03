"""Tests for storage backends."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from src.storage.base import (
    JSONSerializer,
    LocalStorageBackend,
    generate_partition_path,
)
from src.storage.bronze import BronzeStorage


class SampleModel(BaseModel):
    """Sample model for testing."""

    id: str
    name: str
    value: int


class TestJSONSerializer:
    """Tests for JSON serializer."""

    def test_serialize_dict(self):
        """Should serialize dict to JSON bytes."""
        serializer = JSONSerializer()
        data = {"key": "value", "number": 42}

        result = serializer.serialize(data)

        assert isinstance(result, bytes)
        parsed = json.loads(result)
        assert parsed == data

    def test_serialize_pydantic_model(self):
        """Should serialize Pydantic models."""
        serializer = JSONSerializer()
        model = SampleModel(id="1", name="test", value=100)

        result = serializer.serialize(model)

        parsed = json.loads(result)
        assert parsed["id"] == "1"
        assert parsed["name"] == "test"
        assert parsed["value"] == 100

    def test_deserialize_to_model(self):
        """Should deserialize to Pydantic model."""
        serializer = JSONSerializer()
        data = b'{"id": "1", "name": "test", "value": 100}'

        result = serializer.deserialize(data, SampleModel)

        assert isinstance(result, SampleModel)
        assert result.id == "1"
        assert result.name == "test"
        assert result.value == 100


class TestLocalStorageBackend:
    """Tests for local filesystem backend."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def backend(self, temp_dir):
        """Create backend instance."""
        return LocalStorageBackend(base_dir=temp_dir)

    def test_write_and_read(self, backend):
        """Should write and read bytes."""
        content = b"test content"

        path = backend.write(content, "test/file.txt")
        result = backend.read("test/file.txt")

        assert result == content

    def test_creates_directories(self, backend, temp_dir):
        """Should create nested directories."""
        content = b"nested content"

        backend.write(content, "deep/nested/path/file.txt")

        assert (Path(temp_dir) / "deep/nested/path/file.txt").exists()

    def test_exists(self, backend):
        """Should check file existence."""
        backend.write(b"content", "exists.txt")

        assert backend.exists("exists.txt") is True
        assert backend.exists("not_exists.txt") is False

    def test_list_files(self, backend):
        """Should list files matching prefix."""
        backend.write(b"1", "data/2024/01/file1.json")
        backend.write(b"2", "data/2024/01/file2.json")
        backend.write(b"3", "data/2024/02/file3.json")

        files = backend.list_files("data/2024/01")

        assert len(files) == 2
        assert any("file1.json" in f for f in files)
        assert any("file2.json" in f for f in files)

    def test_delete(self, backend):
        """Should delete files."""
        backend.write(b"content", "to_delete.txt")
        assert backend.exists("to_delete.txt")

        result = backend.delete("to_delete.txt")

        assert result is True
        assert backend.exists("to_delete.txt") is False

    def test_delete_nonexistent(self, backend):
        """Should return False for nonexistent files."""
        result = backend.delete("nonexistent.txt")
        assert result is False

    def test_write_model(self, backend):
        """Should write Pydantic model."""
        model = SampleModel(id="1", name="test", value=100)

        backend.write_model(model, "model.json")
        content = backend.read("model.json")

        parsed = json.loads(content)
        assert parsed["id"] == "1"

    def test_read_model(self, backend):
        """Should read and parse Pydantic model."""
        model = SampleModel(id="1", name="test", value=100)
        backend.write_model(model, "model.json")

        result = backend.read_model("model.json", SampleModel)

        assert result.id == "1"
        assert result.name == "test"
        assert result.value == 100


class TestGeneratePartitionPath:
    """Tests for partition path generation."""

    def test_generates_correct_path(self):
        """Should generate partitioned path."""
        timestamp = datetime(2024, 2, 17, 14, 30, 0)

        path = generate_partition_path("matches", "match_123", timestamp)

        assert path == "matches/year=2024/month=02/day=17/match_123.json"

    def test_uses_current_time_if_none(self):
        """Should use current time if no timestamp provided."""
        path = generate_partition_path("matches", "match_123")

        assert "year=" in path
        assert "month=" in path
        assert "day=" in path

    def test_custom_extension(self):
        """Should support custom file extensions."""
        timestamp = datetime(2024, 2, 17)

        path = generate_partition_path("data", "file", timestamp, "parquet")

        assert path.endswith(".parquet")


class TestBronzeStorage:
    """Tests for Bronze layer storage."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def storage(self, temp_dir, monkeypatch):
        """Create Bronze storage with local only."""
        monkeypatch.setenv("STORAGE_LOCAL_BACKUP_DIR", temp_dir)
        return BronzeStorage(enable_s3=False)

    def test_save_dict(self, storage):
        """Should save dictionary data."""
        data = {"key": "value"}

        result = storage.save(data, "test", "item1")

        assert result["local"] is not None
        assert "test" in result["local"]

    def test_save_with_metadata(self, storage):
        """Should include metadata in envelope."""
        data = {"key": "value"}

        result = storage.save(
            data,
            "test",
            "item1",
            metadata={"source": "unit_test"},
        )

        # Read back and verify
        content = Path(result["local"]).read_text()
        envelope = json.loads(content)

        assert "_metadata" in envelope
        assert envelope["_metadata"]["source"] == "unit_test"
        assert envelope["_metadata"]["data_type"] == "test"
        assert "data" in envelope

    def test_save_creates_partitioned_path(self, storage, temp_dir):
        """Should create date-partitioned path."""
        timestamp = datetime(2024, 2, 17)

        storage.save({"key": "value"}, "matches", "m123", timestamp=timestamp)

        expected_path = Path(temp_dir) / "bronze/matches/year=2024/month=02/day=17/m123.json"
        assert expected_path.exists()

    def test_save_batch(self, storage):
        """Should save multiple items."""
        items = [
            ({"id": "1"}, "leagues", "league1"),
            ({"id": "2"}, "leagues", "league2"),
            ({"id": "3"}, "teams", "team1"),
        ]

        results = storage.save_batch(items)

        assert len(results) == 3
        assert all(r["local"] is not None for r in results)

    def test_list_data(self, storage):
        """Should list saved data."""
        timestamp = datetime(2024, 2, 17)
        storage.save({"id": "1"}, "matches", "m1", timestamp=timestamp)
        storage.save({"id": "2"}, "matches", "m2", timestamp=timestamp)
        storage.save({"id": "3"}, "leagues", "l1", timestamp=timestamp)

        match_files = storage.list_data("matches", year=2024, month=2)

        assert len(match_files) == 2
