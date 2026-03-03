"""Shared pytest fixtures and configuration."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_match_data():
    """Sample match data for testing."""
    return {
        "id": "match_123",
        "state": "completed",
        "tournament_id": "tournament_456",
        "league_slug": "cblol",
        "strategy_type": "bestOf",
        "strategy_count": 3,
        "team1": {
            "id": "team_1",
            "name": "Team Alpha",
            "code": "TA",
        },
        "team2": {
            "id": "team_2",
            "name": "Team Beta",
            "code": "TB",
        },
        "games": [
            {"id": "game_1", "state": "finished"},
            {"id": "game_2", "state": "finished"},
            {"id": "game_3", "state": "finished"},
        ],
    }


@pytest.fixture
def sample_league_data():
    """Sample league data for testing."""
    return {
        "id": "league_123",
        "slug": "cblol",
        "name": "CBLOL",
        "region": "BRAZIL",
        "image_url": "https://example.com/cblol.png",
        "priority": 100,
    }


@pytest.fixture
def sample_player_stats():
    """Sample player statistics for testing."""
    return {
        "player_id": "player_1",
        "participant_id": 1,
        "team_id": "team_1",
        "champion_id": 64,  # Lee Sin
        "role": "jungle",
        "side": "blue",
        "kills": 8,
        "deaths": 2,
        "assists": 12,
        "creep_score": 180,
        "gold_earned": 14500,
        "damage_dealt": 22000,
        "vision_score": 45,
    }


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Reset settings cache before each test."""
    from src.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
