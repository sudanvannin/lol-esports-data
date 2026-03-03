"""Tests for ingestion modules."""

import pytest

from src.ingestion.esports_api import DEFAULT_API_KEY, LoLEsportsClient
from src.ingestion.oracle_elixir import (
    AVAILABLE_DATASETS,
    OracleElixirDownloader,
)
from src.models.esports import League, Match, MatchState, Team, Tournament


class TestLoLEsportsClient:
    """Tests for LoL Esports API client."""

    def test_default_api_key_exists(self):
        """API key should be defined."""
        assert DEFAULT_API_KEY is not None
        assert len(DEFAULT_API_KEY) > 0

    def test_league_model(self):
        """League Pydantic model should work correctly."""
        league = League(
            id="123",
            slug="cblol-brazil",
            name="CBLOL",
            region="BRAZIL",
        )
        assert league.id == "123"
        assert league.slug == "cblol-brazil"
        assert league.name == "CBLOL"
        assert league.region == "BRAZIL"
        assert league.image_url is None
        assert league.priority == 0

    def test_tournament_model(self):
        """Tournament Pydantic model should work correctly."""
        tournament = Tournament(
            id="456",
            slug="cblol_2024_split1",
            name="CBLOL 2024 Split 1",
            league_id="123",
        )
        assert tournament.id == "456"
        assert tournament.league_id == "123"
        assert tournament.start_date is None

    def test_match_model(self):
        """Match Pydantic model should work correctly."""
        match = Match(
            id="789",
            state=MatchState.COMPLETED,
            tournament_id="456",
            league_slug="cblol-brazil",
            block_name="Week 1",
            strategy_type="bestOf",
            strategy_count=3,
            team1=Team(id="t1", name="Team A"),
            team2=Team(id="t2", name="Team B"),
        )
        assert match.id == "789"
        assert match.state == MatchState.COMPLETED
        assert match.strategy_count == 3
        assert match.team1.name == "Team A"

    def test_client_initialization(self):
        """Client should initialize with default values."""
        client = LoLEsportsClient()
        assert client.api_key == DEFAULT_API_KEY
        assert client.timeout == 30.0


class TestOracleElixirDownloader:
    """Tests for Oracle's Elixir downloader."""

    def test_available_datasets(self):
        """Should have datasets for multiple years."""
        assert "2024" in AVAILABLE_DATASETS
        assert "2023" in AVAILABLE_DATASETS
        assert "2022" in AVAILABLE_DATASETS
        assert len(AVAILABLE_DATASETS) >= 10

    def test_downloader_initialization(self):
        """Downloader should initialize correctly."""
        downloader = OracleElixirDownloader(output_dir="test_output")
        assert downloader.output_dir.name == "test_output"
        assert downloader.timeout == 300.0

    def test_downloader_custom_timeout(self):
        """Downloader should accept custom timeout."""
        downloader = OracleElixirDownloader(timeout=600.0)
        assert downloader.timeout == 600.0


class TestModelDefaults:
    """Test Pydantic model default values."""

    def test_league_defaults(self):
        """League should have sensible defaults."""
        league = League(id="1", slug="test", name="Test", region="TEST")
        assert league.image_url is None
        assert league.priority == 0

    def test_tournament_defaults(self):
        """Tournament should have sensible defaults."""
        tournament = Tournament(
            id="1", slug="test", name="Test", league_id="1"
        )
        assert tournament.start_date is None
        assert tournament.end_date is None

    def test_match_defaults(self):
        """Match should have sensible defaults."""
        match = Match(
            id="1",
            state=MatchState.COMPLETED,
            tournament_id="1",
        )
        assert match.start_time is None
        assert match.block_name is None
        assert match.team1 is None
        assert match.team2 is None
