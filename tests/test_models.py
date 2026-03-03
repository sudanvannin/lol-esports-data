"""Tests for data models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models.esports import (
    Champion,
    Game,
    GameParticipant,
    GameState,
    GameStats,
    League,
    Match,
    MatchState,
    Player,
    Role,
    Side,
    Team,
    Tournament,
)


class TestLeague:
    """Tests for League model."""

    def test_required_fields(self):
        """Should require id, slug, name."""
        league = League(id="123", slug="cblol", name="CBLOL", region="BRAZIL")
        assert league.id == "123"
        assert league.slug == "cblol"
        assert league.name == "CBLOL"

    def test_optional_fields(self):
        """Should have optional fields with defaults."""
        league = League(id="123", slug="cblol", name="CBLOL", region="BR")
        assert league.image_url is None
        assert league.priority == 0

    def test_validation_strips_whitespace(self):
        """Should strip whitespace from string fields."""
        league = League(id=" 123 ", slug=" cblol ", name=" CBLOL ", region="BR")
        assert league.id == "123"
        assert league.slug == "cblol"


class TestTournament:
    """Tests for Tournament model."""

    def test_date_parsing(self):
        """Should parse ISO date strings."""
        tournament = Tournament(
            id="1",
            slug="split1",
            name="Split 1",
            league_id="123",
            start_date="2024-01-15T00:00:00Z",
        )
        assert isinstance(tournament.start_date, datetime)
        assert tournament.start_date.year == 2024
        assert tournament.start_date.month == 1

    def test_accepts_datetime_objects(self):
        """Should accept datetime objects directly."""
        dt = datetime(2024, 1, 15)
        tournament = Tournament(
            id="1",
            slug="split1",
            name="Split 1",
            league_id="123",
            start_date=dt,
        )
        assert tournament.start_date == dt


class TestTeam:
    """Tests for Team model."""

    def test_minimal_team(self):
        """Should work with minimal fields."""
        team = Team(id="1", name="Test Team")
        assert team.id == "1"
        assert team.name == "Test Team"
        assert team.code == ""

    def test_full_team(self):
        """Should accept all fields."""
        team = Team(
            id="1",
            name="T1",
            code="T1",
            slug="t1",
            image_url="https://example.com/t1.png",
            result="win",
            record={"wins": 10, "losses": 2},
        )
        assert team.code == "T1"
        assert team.record["wins"] == 10


class TestGameParticipant:
    """Tests for GameParticipant model."""

    def test_kda_calculation(self):
        """Should calculate KDA correctly."""
        participant = GameParticipant(
            player_id="p1",
            participant_id=1,
            team_id="t1",
            champion_id=1,
            role=Role.MID,
            side=Side.BLUE,
            kills=10,
            deaths=2,
            assists=5,
        )
        assert participant.kda == 7.5  # (10 + 5) / 2

    def test_kda_zero_deaths(self):
        """Should handle zero deaths in KDA."""
        participant = GameParticipant(
            player_id="p1",
            participant_id=1,
            team_id="t1",
            champion_id=1,
            role=Role.MID,
            side=Side.BLUE,
            kills=5,
            deaths=0,
            assists=10,
        )
        assert participant.kda == 15.0  # 5 + 10

    def test_validates_non_negative(self):
        """Should reject negative values."""
        with pytest.raises(ValidationError):
            GameParticipant(
                player_id="p1",
                participant_id=1,
                team_id="t1",
                champion_id=1,
                role=Role.MID,
                side=Side.BLUE,
                kills=-1,  # Invalid
                deaths=0,
                assists=0,
            )


class TestGame:
    """Tests for Game model."""

    def test_duration_minutes(self):
        """Should calculate duration in minutes."""
        game = Game(
            id="g1",
            match_id="m1",
            game_number=1,
            state=GameState.FINISHED,
            duration_seconds=1800,
        )
        assert game.duration_minutes == 30.0

    def test_duration_minutes_none(self):
        """Should return None if no duration."""
        game = Game(
            id="g1",
            match_id="m1",
            game_number=1,
            state=GameState.UNSTARTED,
        )
        assert game.duration_minutes is None


class TestMatch:
    """Tests for Match model."""

    def test_date_parsing(self):
        """Should parse ISO date strings."""
        match = Match(
            id="m1",
            state=MatchState.COMPLETED,
            tournament_id="t1",
            start_time="2024-02-17T18:00:00Z",
        )
        assert isinstance(match.start_time, datetime)

    def test_state_enum(self):
        """Should accept state as string or enum."""
        match = Match(
            id="m1",
            state="completed",
            tournament_id="t1",
        )
        assert match.state == MatchState.COMPLETED


class TestEnums:
    """Tests for enum types."""

    def test_match_states(self):
        """Should have all match states."""
        assert MatchState.UNSTARTED.value == "unstarted"
        assert MatchState.IN_PROGRESS.value == "inProgress"
        assert MatchState.COMPLETED.value == "completed"

    def test_roles(self):
        """Should have all roles."""
        assert Role.TOP.value == "top"
        assert Role.JUNGLE.value == "jungle"
        assert Role.MID.value == "mid"
        assert Role.BOT.value == "bot"
        assert Role.SUPPORT.value == "support"

    def test_sides(self):
        """Should have both sides."""
        assert Side.BLUE.value == "blue"
        assert Side.RED.value == "red"
