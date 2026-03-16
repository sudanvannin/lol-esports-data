"""Helpers for normalizing upcoming Leaguepedia MatchSchedule rows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DEFAULT_MATCH_SCHEDULE_PATH = Path("data/bronze/leaguepedia/match_results.json")
_UNKNOWN_TEAMS = {"", "tbd", "tba", "unknown"}


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _is_blank(value: object) -> bool:
    return _clean_text(value) == ""


def _looks_unknown_team(value: object) -> bool:
    return _clean_text(value).lower() in _UNKNOWN_TEAMS


def _extract_league_name(overview_page: str, match_id: str) -> str:
    for candidate in (_clean_text(overview_page), _clean_text(match_id).split("_")[0]):
        if not candidate:
            continue
        if "/" in candidate:
            candidate = candidate.split("/")[0].strip()
        if candidate[:4].isdigit() and len(candidate) > 5 and candidate[4] == " ":
            return candidate[5:].strip()
        return candidate
    return "Unknown"


def _build_phase_label(row: dict) -> str:
    parts: list[str] = []
    for key in ("Tab", "Phase", "Round", "MatchDay"):
        value = _clean_text(row.get(key))
        if value and value not in parts:
            parts.append(value)
    return " · ".join(parts)


def normalize_upcoming_match_rows(
    rows: list[dict],
    *,
    reference_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Normalize future unfinished MatchSchedule rows to a compact table."""
    if reference_time is None:
        reference_time = pd.Timestamp.now(tz="UTC")
    else:
        reference_time = pd.Timestamp(reference_time)
        if reference_time.tzinfo is None:
            reference_time = reference_time.tz_localize("UTC")
        else:
            reference_time = reference_time.tz_convert("UTC")

    normalized_rows: list[dict] = []
    for row in rows:
        match_time_raw = row.get("DateTime UTC", row.get("DateTime_UTC"))
        match_time = pd.to_datetime(match_time_raw, utc=True, errors="coerce")
        if pd.isna(match_time) or match_time < reference_time:
            continue

        team1 = _clean_text(row.get("Team1"))
        team2 = _clean_text(row.get("Team2"))
        if _looks_unknown_team(team1) or _looks_unknown_team(team2):
            continue

        # Upcoming only: no declared winner and no resolved series score yet.
        if not _is_blank(row.get("Winner")):
            continue
        if not _is_blank(row.get("Team1Score")) or not _is_blank(row.get("Team2Score")):
            continue

        overview_page = _clean_text(row.get("OverviewPage"))
        match_id = _clean_text(row.get("MatchId"))
        normalized_rows.append(
            {
                "match_id": match_id,
                "match_time": match_time,
                "match_date": match_time.floor("D"),
                "league": _extract_league_name(overview_page, match_id),
                "event_name": overview_page or match_id,
                "phase_label": _build_phase_label(row),
                "team1": team1,
                "team2": team2,
                "best_of": pd.to_numeric(
                    _clean_text(row.get("BestOf")), errors="coerce"
                ),
                "patch": _clean_text(row.get("Patch")),
                "overview_page": overview_page,
                "source": "leaguepedia_matchschedule",
            }
        )

    if not normalized_rows:
        return pd.DataFrame(
            columns=[
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
        )

    upcoming_df = pd.DataFrame(normalized_rows)
    if "best_of" in upcoming_df.columns:
        upcoming_df["best_of"] = upcoming_df["best_of"].astype("Int64")
    return upcoming_df.sort_values(
        ["match_time", "league", "team1", "team2"], kind="stable"
    ).reset_index(drop=True)


def load_upcoming_matches(
    path: Path = DEFAULT_MATCH_SCHEDULE_PATH,
    *,
    reference_time: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load the Leaguepedia schedule payload and return normalized upcoming rows plus metadata."""
    if not path.exists():
        return normalize_upcoming_match_rows([], reference_time=reference_time), {
            "path": str(path),
            "fetched_at": None,
            "row_count": 0,
            "completed_years": [],
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("rows", [])
        metadata = {
            "path": str(path),
            "fetched_at": payload.get("fetched_at"),
            "row_count": int(payload.get("row_count", len(rows))),
            "completed_years": payload.get("completed_years", []),
        }
    elif isinstance(payload, list):
        rows = payload
        metadata = {
            "path": str(path),
            "fetched_at": None,
            "row_count": len(rows),
            "completed_years": [],
        }
    else:
        rows = []
        metadata = {
            "path": str(path),
            "fetched_at": None,
            "row_count": 0,
            "completed_years": [],
        }

    return normalize_upcoming_match_rows(rows, reference_time=reference_time), metadata
