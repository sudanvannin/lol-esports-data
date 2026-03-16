"""Helpers for building a fast web schedule snapshot from the official Riot API."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion.esports_api import LoLEsportsClient

DEFAULT_WEB_SCHEDULE_PATH = Path("data/bronze/official/web_schedule.json")

RECENT_MATCH_COLUMNS = [
    "match_id",
    "match_time",
    "match_date",
    "league",
    "event_name",
    "tournament_phase",
    "team1",
    "team2",
    "team1_wins",
    "team2_wins",
    "score",
    "series_winner",
    "series_format",
    "best_of",
    "state",
    "source",
]

UPCOMING_MATCH_COLUMNS = [
    "match_id",
    "match_time",
    "match_date",
    "league",
    "event_name",
    "phase_label",
    "team1",
    "team2",
    "best_of",
    "patch",
    "overview_page",
    "source",
]

_UNKNOWN_TEAMS = {"", "tbd", "tba", "unknown"}


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _is_unknown_team(name: object) -> bool:
    return _clean_text(name).lower() in _UNKNOWN_TEAMS


def _parse_timestamp(value: object) -> pd.Timestamp | pd.NaT:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    return timestamp


def _team_name(team: dict[str, Any] | None) -> str:
    if not team:
        return ""
    for key in ("name", "code", "slug"):
        text = _clean_text(team.get(key))
        if text:
            return text
    return ""


def _team_id(team: dict[str, Any] | None) -> str:
    if not team:
        return ""
    return _clean_text(team.get("id"))


def _team_outcome(team: dict[str, Any] | None) -> str:
    if not team:
        return ""
    result = team.get("result")
    if isinstance(result, dict):
        return _clean_text(result.get("outcome")).lower()
    return _clean_text(team.get("outcome")).lower()


def _team_wins(team: dict[str, Any] | None) -> int | None:
    if not team:
        return None

    candidates = []
    result = team.get("result")
    if isinstance(result, dict):
        candidates.extend(
            [
                result.get("gameWins"),
                result.get("wins"),
                result.get("score"),
            ]
        )
    record = team.get("record")
    if isinstance(record, dict):
        candidates.extend([record.get("wins"), record.get("score")])
    candidates.extend([team.get("gameWins"), team.get("wins"), team.get("score")])

    for candidate in candidates:
        wins = pd.to_numeric(candidate, errors="coerce")
        if pd.notna(wins):
            return int(wins)
    return None


def _winner_id_from_game(game: dict[str, Any] | None) -> str:
    if not game:
        return ""

    winner = game.get("winner")
    if isinstance(winner, dict):
        winner_id = _clean_text(winner.get("id"))
        if winner_id:
            return winner_id

    for key in ("winnerId", "winningTeamId", "teamId"):
        winner_id = _clean_text(game.get(key))
        if winner_id:
            return winner_id

    teams = game.get("teams")
    if isinstance(teams, list):
        for team in teams:
            outcome = _team_outcome(team)
            if outcome in {"win", "won"}:
                winner_id = _team_id(team)
                if winner_id:
                    return winner_id
    return ""


def _score_from_games(
    games: list[dict[str, Any]],
    team1_id: str,
    team2_id: str,
) -> tuple[int | None, int | None]:
    if not games:
        return None, None

    team1_wins = 0
    team2_wins = 0

    for game in games:
        winner_id = _winner_id_from_game(game)
        if winner_id and winner_id == team1_id:
            team1_wins += 1
        elif winner_id and winner_id == team2_id:
            team2_wins += 1

    if team1_wins == 0 and team2_wins == 0:
        return None, None
    return team1_wins, team2_wins


def _score_from_teams_or_games(
    team1: dict[str, Any],
    team2: dict[str, Any],
    games: list[dict[str, Any]],
) -> tuple[int | None, int | None]:
    team1_wins = _team_wins(team1)
    team2_wins = _team_wins(team2)
    if team1_wins is not None or team2_wins is not None:
        return team1_wins or 0, team2_wins or 0
    return _score_from_games(games, _team_id(team1), _team_id(team2))


def _series_winner_name(
    team1: dict[str, Any],
    team2: dict[str, Any],
    team1_wins: int | None,
    team2_wins: int | None,
) -> str | None:
    outcome1 = _team_outcome(team1)
    outcome2 = _team_outcome(team2)

    if outcome1 in {"win", "won"}:
        return _team_name(team1)
    if outcome2 in {"win", "won"}:
        return _team_name(team2)

    if team1_wins is not None and team2_wins is not None:
        if team1_wins > team2_wins:
            return _team_name(team1)
        if team2_wins > team1_wins:
            return _team_name(team2)
    return None


def _event_name(event: dict[str, Any]) -> str:
    tournament = event.get("tournament")
    if isinstance(tournament, dict):
        name = _clean_text(tournament.get("name"))
        if name:
            return name

    league = event.get("league")
    if isinstance(league, dict):
        name = _clean_text(league.get("name"))
        if name:
            return name

    return _clean_text(event.get("blockName")) or "Unknown event"


def _league_name(event: dict[str, Any]) -> str:
    league = event.get("league")
    if isinstance(league, dict):
        name = _clean_text(league.get("name"))
        if name:
            return name
        slug = _clean_text(league.get("slug"))
        if slug:
            return slug.upper()
    return "Unknown"


def _phase_label(event: dict[str, Any]) -> str:
    return _clean_text(event.get("blockName"))


def _match_best_of(match: dict[str, Any]) -> int | None:
    strategy = match.get("strategy")
    if isinstance(strategy, dict):
        count = pd.to_numeric(strategy.get("count"), errors="coerce")
        if pd.notna(count):
            return int(count)

    games = match.get("games")
    if isinstance(games, list) and games:
        return max(len(games), 1)
    return None


def _patch_from_match(match: dict[str, Any]) -> str | None:
    games = match.get("games")
    if not isinstance(games, list):
        return None

    for game in games:
        for key in ("patch", "gameVersion"):
            patch = _clean_text(game.get(key))
            if patch:
                return patch
    return None


def normalize_schedule_events(
    events: list[dict[str, Any]],
    *,
    reference_time: pd.Timestamp | None = None,
    recent_lookback_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize official Riot schedule events into recent and upcoming tables."""
    if reference_time is None:
        reference_time = pd.Timestamp.now(tz="UTC")
    else:
        reference_time = pd.Timestamp(reference_time)
        if reference_time.tzinfo is None:
            reference_time = reference_time.tz_localize("UTC")
        else:
            reference_time = reference_time.tz_convert("UTC")

    recent_cutoff = reference_time - pd.Timedelta(days=recent_lookback_days)

    recent_rows: list[dict[str, Any]] = []
    upcoming_rows: list[dict[str, Any]] = []

    for event in events:
        if _clean_text(event.get("type")).lower() not in {"match", ""}:
            continue

        match = event.get("match")
        if not isinstance(match, dict):
            continue

        teams = match.get("teams")
        if not isinstance(teams, list) or len(teams) < 2:
            continue

        team1 = teams[0] or {}
        team2 = teams[1] or {}
        team1_name = _team_name(team1)
        team2_name = _team_name(team2)
        if _is_unknown_team(team1_name) or _is_unknown_team(team2_name):
            continue

        match_time = _parse_timestamp(event.get("startTime"))
        if pd.isna(match_time):
            continue

        match_id = _clean_text(match.get("id")) or _clean_text(event.get("id"))
        league_name = _league_name(event)
        event_name = _event_name(event)
        phase_label = _phase_label(event)
        best_of = _match_best_of(match)
        state = _clean_text(event.get("state"))

        if state == "completed" and match_time >= recent_cutoff:
            games = match.get("games") if isinstance(match.get("games"), list) else []
            team1_wins, team2_wins = _score_from_teams_or_games(team1, team2, games)
            score = None
            if team1_wins is not None and team2_wins is not None:
                score = f"{team1_wins}-{team2_wins}"

            recent_rows.append(
                {
                    "match_id": match_id,
                    "match_time": match_time,
                    "match_date": match_time,
                    "league": league_name,
                    "event_name": event_name,
                    "tournament_phase": phase_label,
                    "team1": team1_name,
                    "team2": team2_name,
                    "team1_wins": team1_wins,
                    "team2_wins": team2_wins,
                    "score": score,
                    "series_winner": _series_winner_name(
                        team1, team2, team1_wins, team2_wins
                    ),
                    "series_format": f"Bo{best_of}" if best_of else None,
                    "best_of": best_of,
                    "state": state,
                    "source": "riot_official_schedule",
                }
            )

        if state in {"unstarted", "inProgress"} and match_time >= reference_time:
            upcoming_rows.append(
                {
                    "match_id": match_id,
                    "match_time": match_time,
                    "match_date": match_time.floor("D"),
                    "league": league_name,
                    "event_name": event_name,
                    "phase_label": phase_label,
                    "team1": team1_name,
                    "team2": team2_name,
                    "best_of": best_of,
                    "patch": _patch_from_match(match),
                    "overview_page": None,
                    "source": "riot_official_schedule",
                }
            )

    recent_df = (
        pd.DataFrame(recent_rows, columns=RECENT_MATCH_COLUMNS)
        if recent_rows
        else pd.DataFrame(columns=RECENT_MATCH_COLUMNS)
    )
    if not recent_df.empty and "best_of" in recent_df.columns:
        recent_df["best_of"] = pd.to_numeric(
            recent_df["best_of"], errors="coerce"
        ).astype("Int64")
        recent_df["team1_wins"] = pd.to_numeric(
            recent_df["team1_wins"], errors="coerce"
        ).astype("Int64")
        recent_df["team2_wins"] = pd.to_numeric(
            recent_df["team2_wins"], errors="coerce"
        ).astype("Int64")
        recent_df = recent_df.sort_values(
            ["match_time", "league", "team1", "team2"],
            ascending=[False, True, True, True],
            kind="stable",
        ).reset_index(drop=True)

    upcoming_df = (
        pd.DataFrame(upcoming_rows, columns=UPCOMING_MATCH_COLUMNS)
        if upcoming_rows
        else pd.DataFrame(columns=UPCOMING_MATCH_COLUMNS)
    )
    if not upcoming_df.empty and "best_of" in upcoming_df.columns:
        upcoming_df["best_of"] = pd.to_numeric(
            upcoming_df["best_of"], errors="coerce"
        ).astype("Int64")
        upcoming_df = upcoming_df.sort_values(
            ["match_time", "league", "team1", "team2"],
            kind="stable",
        ).reset_index(drop=True)

    return recent_df, upcoming_df


async def collect_official_schedule_snapshot(
    *,
    max_pages: int = 3,
    recent_limit: int = 80,
    upcoming_limit: int = 80,
    recent_lookback_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fetch enough schedule pages to build a compact recent/upcoming snapshot."""
    reference_time = pd.Timestamp.now(tz="UTC")
    recent_cutoff = reference_time - pd.Timedelta(days=recent_lookback_days)

    events: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    page_token: str | None = None
    pages_fetched = 0

    async with LoLEsportsClient() as client:
        while pages_fetched < max_pages:
            payload = await client.get_schedule(page_token=page_token)
            schedule = payload.get("schedule", {})
            page_events = schedule.get("events", [])
            pages_fetched += 1

            oldest_event_time: pd.Timestamp | None = None
            for event in page_events:
                event_id = _clean_text(event.get("id"))
                if event_id and event_id in seen_event_ids:
                    continue
                if event_id:
                    seen_event_ids.add(event_id)
                events.append(event)

                event_time = _parse_timestamp(event.get("startTime"))
                if pd.notna(event_time):
                    if oldest_event_time is None or event_time < oldest_event_time:
                        oldest_event_time = event_time

            recent_df, upcoming_df = normalize_schedule_events(
                events,
                reference_time=reference_time,
                recent_lookback_days=recent_lookback_days,
            )

            enough_recent = len(recent_df) >= recent_limit
            enough_upcoming = len(upcoming_df) >= upcoming_limit
            if enough_recent and enough_upcoming:
                break

            older_token = schedule.get("pages", {}).get("older")
            if not older_token:
                break

            if oldest_event_time is not None and oldest_event_time < recent_cutoff:
                break

            page_token = older_token

    recent_df, upcoming_df = normalize_schedule_events(
        events,
        reference_time=reference_time,
        recent_lookback_days=recent_lookback_days,
    )
    recent_df = recent_df.head(recent_limit).reset_index(drop=True)
    upcoming_df = upcoming_df.head(upcoming_limit).reset_index(drop=True)

    metadata = {
        "fetched_at": reference_time.isoformat(),
        "event_count": len(events),
        "pages_fetched": pages_fetched,
        "recent_row_count": int(len(recent_df)),
        "upcoming_row_count": int(len(upcoming_df)),
        "recent_lookback_days": int(recent_lookback_days),
        "path": str(DEFAULT_WEB_SCHEDULE_PATH),
        "source": "riot_official_schedule",
    }
    return recent_df, upcoming_df, metadata


def _records_for_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []

    serializable = df.copy()
    for column in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[column]):
            serializable[column] = serializable[column].dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        elif str(serializable[column].dtype) == "Int64":
            serializable[column] = serializable[column].astype("Int64").astype(object)
    return serializable.where(pd.notna(serializable), None).to_dict("records")


def save_official_schedule_snapshot(
    recent_df: pd.DataFrame,
    upcoming_df: pd.DataFrame,
    metadata: dict[str, Any],
    path: Path = DEFAULT_WEB_SCHEDULE_PATH,
) -> Path:
    """Persist the compact web schedule snapshot for local fallback and inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **metadata,
        "recent_matches": _records_for_json(recent_df),
        "upcoming_matches": _records_for_json(upcoming_df),
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def load_official_schedule_snapshot(
    path: Path = DEFAULT_WEB_SCHEDULE_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the saved compact schedule snapshot from disk."""
    if not path.exists():
        return (
            pd.DataFrame(columns=RECENT_MATCH_COLUMNS),
            pd.DataFrame(columns=UPCOMING_MATCH_COLUMNS),
            {
                "path": str(path),
                "fetched_at": None,
                "event_count": 0,
                "pages_fetched": 0,
                "recent_row_count": 0,
                "upcoming_row_count": 0,
                "recent_lookback_days": 0,
                "source": "riot_official_schedule",
            },
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    recent_df = pd.DataFrame(
        payload.get("recent_matches", []), columns=RECENT_MATCH_COLUMNS
    )
    upcoming_df = pd.DataFrame(
        payload.get("upcoming_matches", []), columns=UPCOMING_MATCH_COLUMNS
    )

    for frame in (recent_df, upcoming_df):
        for column in ("match_time", "match_date"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if "best_of" in frame.columns:
            frame["best_of"] = pd.to_numeric(frame["best_of"], errors="coerce").astype(
                "Int64"
            )

    if "team1_wins" in recent_df.columns:
        recent_df["team1_wins"] = pd.to_numeric(
            recent_df["team1_wins"], errors="coerce"
        ).astype("Int64")
    if "team2_wins" in recent_df.columns:
        recent_df["team2_wins"] = pd.to_numeric(
            recent_df["team2_wins"], errors="coerce"
        ).astype("Int64")

    metadata = {
        "path": str(path),
        "fetched_at": payload.get("fetched_at"),
        "event_count": int(payload.get("event_count", 0)),
        "pages_fetched": int(payload.get("pages_fetched", 0)),
        "recent_row_count": int(payload.get("recent_row_count", len(recent_df))),
        "upcoming_row_count": int(payload.get("upcoming_row_count", len(upcoming_df))),
        "recent_lookback_days": int(payload.get("recent_lookback_days", 0)),
        "source": payload.get("source", "riot_official_schedule"),
    }
    return recent_df, upcoming_df, metadata


def build_and_save_official_schedule_snapshot(
    *,
    path: Path = DEFAULT_WEB_SCHEDULE_PATH,
    max_pages: int = 3,
    recent_limit: int = 80,
    upcoming_limit: int = 80,
    recent_lookback_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Convenience wrapper for scripts."""
    recent_df, upcoming_df, metadata = asyncio.run(
        collect_official_schedule_snapshot(
            max_pages=max_pages,
            recent_limit=recent_limit,
            upcoming_limit=upcoming_limit,
            recent_lookback_days=recent_lookback_days,
        )
    )
    save_official_schedule_snapshot(recent_df, upcoming_df, metadata, path=path)
    return recent_df, upcoming_df, metadata
