"""Official schedule helpers for the web app.

Prefer ``src.official_schedule`` when present. Keep a local fallback loader so
the web layer can still parse a saved snapshot without importing the full src
tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from src.official_schedule import (
        DEFAULT_WEB_SCHEDULE_PATH,
        RECENT_MATCH_COLUMNS,
        UPCOMING_MATCH_COLUMNS,
        load_official_schedule_snapshot,
    )
except ModuleNotFoundError:
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

    def load_official_schedule_snapshot(
        path: Path = DEFAULT_WEB_SCHEDULE_PATH,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
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
            payload.get("recent_matches", []),
            columns=RECENT_MATCH_COLUMNS,
        )
        upcoming_df = pd.DataFrame(
            payload.get("upcoming_matches", []),
            columns=UPCOMING_MATCH_COLUMNS,
        )

        for frame in (recent_df, upcoming_df):
            for column in ("match_time", "match_date"):
                if column in frame.columns:
                    frame[column] = pd.to_datetime(
                        frame[column], utc=True, errors="coerce"
                    )
            if "best_of" in frame.columns:
                frame["best_of"] = pd.to_numeric(
                    frame["best_of"], errors="coerce"
                ).astype("Int64")

        for column in ("team1_wins", "team2_wins"):
            if column in recent_df.columns:
                recent_df[column] = pd.to_numeric(
                    recent_df[column], errors="coerce"
                ).astype("Int64")

        metadata = {
            "path": str(path),
            "fetched_at": payload.get("fetched_at"),
            "event_count": int(payload.get("event_count", 0)),
            "pages_fetched": int(payload.get("pages_fetched", 0)),
            "recent_row_count": int(payload.get("recent_row_count", len(recent_df))),
            "upcoming_row_count": int(
                payload.get("upcoming_row_count", len(upcoming_df))
            ),
            "recent_lookback_days": int(payload.get("recent_lookback_days", 0)),
            "source": payload.get("source", "riot_official_schedule"),
        }
        return recent_df, upcoming_df, metadata
