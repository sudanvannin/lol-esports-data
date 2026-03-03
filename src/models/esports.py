"""
Pydantic models for LoL Esports data.

These models define the schema for data flowing through the pipeline.
They provide validation, serialization, and documentation.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class MatchState(str, Enum):
    """Possible states for a match."""

    UNSTARTED = "unstarted"
    IN_PROGRESS = "inProgress"
    COMPLETED = "completed"


class GameState(str, Enum):
    """Possible states for a game."""

    UNSTARTED = "unstarted"
    IN_PROGRESS = "inProgress"
    FINISHED = "finished"


class Side(str, Enum):
    """Team side in a game."""

    BLUE = "blue"
    RED = "red"


class Role(str, Enum):
    """Player role/position."""

    TOP = "top"
    JUNGLE = "jungle"
    MID = "mid"
    BOT = "bot"
    SUPPORT = "support"


class League(BaseModel):
    """Represents a competitive league (e.g., LCK, LEC, CBLOL)."""

    id: str = Field(..., description="Unique league identifier")
    slug: str = Field(..., description="URL-friendly slug")
    name: str = Field(..., description="Display name")
    region: str = Field(default="", description="Geographic region")
    image_url: str | None = Field(default=None, description="League logo URL")
    priority: int = Field(default=0, description="Display priority")

    class Config:
        str_strip_whitespace = True


class Tournament(BaseModel):
    """Represents a tournament within a league (e.g., Split 1, Playoffs)."""

    id: str = Field(..., description="Unique tournament identifier")
    slug: str = Field(..., description="URL-friendly slug")
    name: str = Field(..., description="Display name")
    league_id: str = Field(..., description="Parent league ID")
    start_date: datetime | None = Field(default=None, description="Tournament start")
    end_date: datetime | None = Field(default=None, description="Tournament end")

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class Team(BaseModel):
    """Represents a professional team."""

    id: str = Field(..., description="Unique team identifier")
    name: str = Field(..., description="Team name")
    code: str = Field(default="", description="Short code (e.g., T1, GEN)")
    slug: str = Field(default="", description="URL-friendly slug")
    image_url: str | None = Field(default=None, description="Team logo URL")
    result: str | None = Field(default=None, description="Match result (win/loss)")
    record: dict[str, int] | None = Field(default=None, description="Win/loss record")


class Player(BaseModel):
    """Represents a professional player."""

    id: str = Field(..., description="Unique player identifier")
    summoner_name: str = Field(..., description="In-game name")
    first_name: str = Field(default="", description="Real first name")
    last_name: str = Field(default="", description="Real last name")
    role: Role | str = Field(..., description="Player role")
    team_id: str | None = Field(default=None, description="Current team ID")
    image_url: str | None = Field(default=None, description="Player photo URL")


class Champion(BaseModel):
    """Represents a League of Legends champion."""

    id: int = Field(..., description="Champion ID")
    name: str = Field(..., description="Champion name")
    image_url: str | None = Field(default=None, description="Champion image URL")


class GameParticipant(BaseModel):
    """Player statistics for a single game."""

    player_id: str = Field(..., description="Player identifier")
    participant_id: int = Field(..., description="In-game participant ID (1-10)")
    team_id: str = Field(..., description="Team identifier")
    champion_id: int = Field(..., description="Champion ID")
    role: Role | str = Field(..., description="Role played")
    side: Side = Field(..., description="Blue or Red side")

    kills: int = Field(default=0, ge=0)
    deaths: int = Field(default=0, ge=0)
    assists: int = Field(default=0, ge=0)
    creep_score: int = Field(default=0, ge=0, description="Total CS")
    gold_earned: int = Field(default=0, ge=0)
    damage_dealt: int = Field(default=0, ge=0, description="Damage to champions")
    vision_score: int = Field(default=0, ge=0)
    wards_placed: int = Field(default=0, ge=0)
    wards_killed: int = Field(default=0, ge=0)

    items: list[int] = Field(default_factory=list, description="Item IDs")
    summoner_spells: list[int] = Field(default_factory=list, description="Summoner spell IDs")
    runes: dict[str, Any] = Field(default_factory=dict, description="Rune selection")

    @property
    def kda(self) -> float:
        """Calculate KDA ratio."""
        if self.deaths == 0:
            return float(self.kills + self.assists)
        return (self.kills + self.assists) / self.deaths

    @property
    def cs_per_minute(self) -> float:
        """Calculate CS per minute (requires game duration context)."""
        return 0.0  # Needs game context


class GameStats(BaseModel):
    """Team statistics for a single game."""

    team_id: str = Field(..., description="Team identifier")
    side: Side = Field(..., description="Blue or Red side")
    is_winner: bool = Field(..., description="Whether team won")

    kills: int = Field(default=0, ge=0)
    deaths: int = Field(default=0, ge=0)
    assists: int = Field(default=0, ge=0)
    gold: int = Field(default=0, ge=0)
    towers: int = Field(default=0, ge=0)
    dragons: int = Field(default=0, ge=0)
    barons: int = Field(default=0, ge=0)
    heralds: int = Field(default=0, ge=0)
    inhibitors: int = Field(default=0, ge=0)

    first_blood: bool = Field(default=False)
    first_tower: bool = Field(default=False)
    first_dragon: bool = Field(default=False)
    first_baron: bool = Field(default=False)

    bans: list[int] = Field(default_factory=list, description="Banned champion IDs")

    gold_at_10: int | None = Field(default=None, description="Gold at 10 minutes")
    gold_at_15: int | None = Field(default=None, description="Gold at 15 minutes")


class Game(BaseModel):
    """Represents a single game within a match."""

    id: str = Field(..., description="Unique game identifier")
    match_id: str = Field(..., description="Parent match ID")
    game_number: int = Field(..., ge=1, description="Game number in the series")
    state: GameState = Field(..., description="Current game state")
    patch: str = Field(default="", description="Game patch version")
    duration_seconds: int | None = Field(default=None, ge=0, description="Game duration")

    blue_team: GameStats | None = Field(default=None, description="Blue side stats")
    red_team: GameStats | None = Field(default=None, description="Red side stats")
    participants: list[GameParticipant] = Field(default_factory=list)

    vod_url: str | None = Field(default=None, description="VOD link")
    platform_game_id: str | None = Field(default=None, description="Platform-specific game ID")

    @property
    def duration_minutes(self) -> float | None:
        """Game duration in minutes."""
        if self.duration_seconds:
            return self.duration_seconds / 60
        return None


class Match(BaseModel):
    """Represents a match (series) between two teams."""

    id: str = Field(..., description="Unique match identifier")
    state: MatchState = Field(..., description="Current match state")
    tournament_id: str = Field(..., description="Parent tournament ID")
    league_slug: str = Field(default="", description="League slug")

    block_name: str | None = Field(default=None, description="Block name (e.g., Week 1)")
    strategy_type: str = Field(default="bestOf", description="Match type")
    strategy_count: int = Field(default=1, ge=1, description="Best of N")

    team1: Team | None = Field(default=None, description="First team")
    team2: Team | None = Field(default=None, description="Second team")
    winner_id: str | None = Field(default=None, description="Winning team ID")

    games: list[Game] = Field(default_factory=list)
    start_time: datetime | None = Field(default=None, description="Scheduled start time")

    @field_validator("start_time", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    @property
    def score(self) -> tuple[int, int] | None:
        """Get match score as (team1_wins, team2_wins)."""
        if not self.team1 or not self.team2 or not self.games:
            return None

        t1_wins = sum(
            1 for g in self.games
            if g.blue_team and g.blue_team.team_id == self.team1.id and g.blue_team.is_winner
            or g.red_team and g.red_team.team_id == self.team1.id and g.red_team.is_winner
        )
        t2_wins = len([g for g in self.games if g.state == GameState.FINISHED]) - t1_wins
        return (t1_wins, t2_wins)


class LiveFrame(BaseModel):
    """A frame of live game data (snapshot at a point in time)."""

    timestamp: datetime = Field(..., description="Frame timestamp")
    game_time_ms: int = Field(..., ge=0, description="In-game time in milliseconds")

    blue_team_gold: int = Field(default=0, ge=0)
    red_team_gold: int = Field(default=0, ge=0)
    blue_team_kills: int = Field(default=0, ge=0)
    red_team_kills: int = Field(default=0, ge=0)

    participants: list[dict] = Field(default_factory=list, description="Participant snapshots")

    @property
    def gold_diff(self) -> int:
        """Gold difference (blue - red)."""
        return self.blue_team_gold - self.red_team_gold


class GameEvent(BaseModel):
    """An event that occurred during a game."""

    timestamp: datetime = Field(..., description="Event timestamp")
    game_time_ms: int = Field(..., ge=0, description="In-game time")
    event_type: str = Field(..., description="Type of event")

    killer_id: int | None = Field(default=None, description="Killer participant ID")
    victim_id: int | None = Field(default=None, description="Victim participant ID")
    assistants: list[int] = Field(default_factory=list, description="Assistant participant IDs")

    position_x: float | None = Field(default=None, description="X coordinate")
    position_y: float | None = Field(default=None, description="Y coordinate")

    details: dict[str, Any] = Field(default_factory=dict, description="Event-specific details")
