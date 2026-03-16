"""Tests for official Riot schedule normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.official_schedule import (
    load_official_schedule_snapshot,
    normalize_schedule_events,
    save_official_schedule_snapshot,
)


def test_normalize_schedule_events_splits_completed_and_upcoming():
    events = [
        {
            "id": "evt_completed",
            "type": "match",
            "state": "completed",
            "startTime": "2026-03-15T13:00:00Z",
            "blockName": "Round 1",
            "league": {"name": "LCK"},
            "tournament": {"name": "First Stand 2026"},
            "match": {
                "id": "match_completed",
                "strategy": {"count": 5},
                "teams": [
                    {
                        "id": "blg",
                        "name": "Bilibili Gaming",
                        "result": {"outcome": "win", "gameWins": 3},
                    },
                    {
                        "id": "fx",
                        "name": "BNK FEARX",
                        "result": {"outcome": "loss", "gameWins": 1},
                    },
                ],
                "games": [
                    {"id": "g1", "winner": {"id": "blg"}},
                    {"id": "g2", "winner": {"id": "blg"}},
                    {"id": "g3", "winner": {"id": "fx"}},
                    {"id": "g4", "winner": {"id": "blg"}},
                ],
            },
        },
        {
            "id": "evt_upcoming",
            "type": "match",
            "state": "unstarted",
            "startTime": "2026-03-16T13:00:00Z",
            "blockName": "Round 2",
            "league": {"name": "First Stand"},
            "tournament": {"name": "First Stand 2026"},
            "match": {
                "id": "match_upcoming",
                "strategy": {"count": 3},
                "teams": [
                    {"id": "g2", "name": "G2 Esports"},
                    {"id": "tsw", "name": "Team Secret Whales"},
                ],
                "games": [{"id": "g1", "patch": "26.05"}],
            },
        },
        {
            "id": "evt_unknown",
            "type": "match",
            "state": "unstarted",
            "startTime": "2026-03-16T14:00:00Z",
            "league": {"name": "LCK"},
            "tournament": {"name": "Ignored"},
            "match": {
                "id": "match_unknown",
                "teams": [
                    {"id": "tbd", "name": "TBD"},
                    {"id": "ok", "name": "Real Team"},
                ],
            },
        },
        {
            "id": "evt_old",
            "type": "match",
            "state": "completed",
            "startTime": "2026-03-01T13:00:00Z",
            "league": {"name": "LPL"},
            "tournament": {"name": "Old Event"},
            "match": {
                "id": "match_old",
                "teams": [
                    {"id": "a", "name": "Team A", "result": {"gameWins": 2}},
                    {"id": "b", "name": "Team B", "result": {"gameWins": 0}},
                ],
            },
        },
    ]

    recent_df, upcoming_df = normalize_schedule_events(
        events,
        reference_time=pd.Timestamp("2026-03-16T00:00:00Z"),
        recent_lookback_days=7,
    )

    assert len(recent_df) == 1
    assert len(upcoming_df) == 1

    recent = recent_df.iloc[0].to_dict()
    assert recent["match_id"] == "match_completed"
    assert recent["league"] == "LCK"
    assert recent["event_name"] == "First Stand 2026"
    assert recent["tournament_phase"] == "Round 1"
    assert recent["score"] == "3-1"
    assert recent["series_winner"] == "Bilibili Gaming"
    assert recent["series_format"] == "Bo5"

    upcoming = upcoming_df.iloc[0].to_dict()
    assert upcoming["match_id"] == "match_upcoming"
    assert upcoming["team1"] == "G2 Esports"
    assert upcoming["team2"] == "Team Secret Whales"
    assert upcoming["patch"] == "26.05"
    assert upcoming["best_of"] == 3


def test_official_schedule_snapshot_round_trip(monkeypatch: pytest.MonkeyPatch):
    recent_df = pd.DataFrame(
        [
            {
                "match_id": "recent_1",
                "match_time": pd.Timestamp("2026-03-16T12:00:00Z"),
                "match_date": pd.Timestamp("2026-03-16T12:00:00Z"),
                "league": "LCK",
                "event_name": "LCK Cup 2026",
                "tournament_phase": "Week 1",
                "team1": "T1",
                "team2": "Gen.G",
                "team1_wins": 2,
                "team2_wins": 1,
                "score": "2-1",
                "series_winner": "T1",
                "series_format": "Bo3",
                "best_of": 3,
                "state": "completed",
                "source": "riot_official_schedule",
            }
        ]
    )
    upcoming_df = pd.DataFrame(
        [
            {
                "match_id": "upcoming_1",
                "match_time": pd.Timestamp("2026-03-17T12:00:00Z"),
                "match_date": pd.Timestamp("2026-03-17T00:00:00Z"),
                "league": "First Stand",
                "event_name": "First Stand 2026",
                "phase_label": "Round 2",
                "team1": "G2 Esports",
                "team2": "T1",
                "best_of": 5,
                "patch": "26.05",
                "overview_page": None,
                "source": "riot_official_schedule",
            }
        ]
    )
    metadata = {
        "fetched_at": "2026-03-16T10:00:00Z",
        "event_count": 20,
        "pages_fetched": 2,
        "recent_row_count": 1,
        "upcoming_row_count": 1,
        "recent_lookback_days": 7,
        "path": "",
        "source": "riot_official_schedule",
    }

    snapshot_path = Path("data/test_runs/official_schedule_roundtrip.json")
    metadata["path"] = str(snapshot_path)
    captured: dict[str, str] = {}

    def fake_write_text(self: Path, text: str, encoding: str = "utf-8") -> int:
        captured["text"] = text
        return len(text)

    monkeypatch.setattr(Path, "write_text", fake_write_text)
    monkeypatch.setattr(Path, "exists", lambda self: self == snapshot_path)
    monkeypatch.setattr(
        Path, "read_text", lambda self, encoding="utf-8": captured["text"]
    )

    save_official_schedule_snapshot(recent_df, upcoming_df, metadata, snapshot_path)
    payload = json.loads(captured["text"])
    loaded_recent, loaded_upcoming, loaded_meta = load_official_schedule_snapshot(
        snapshot_path
    )

    assert payload["event_count"] == 20
    assert len(payload["recent_matches"]) == 1
    assert len(loaded_recent) == 1
    assert len(loaded_upcoming) == 1
    assert loaded_recent.iloc[0]["score"] == "2-1"
    assert loaded_upcoming.iloc[0]["team2"] == "T1"
    assert loaded_meta["event_count"] == 20
    assert loaded_meta["source"] == "riot_official_schedule"
