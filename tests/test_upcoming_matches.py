"""Tests for upcoming match normalization."""

from __future__ import annotations

import pandas as pd

from src.upcoming_matches import normalize_upcoming_match_rows


def test_normalize_upcoming_match_rows_keeps_only_future_unfinished_known_teams():
    rows = [
        {
            "MatchId": "First Stand Tournament/2026_Round 1_1",
            "Team1": "Bilibili Gaming",
            "Team2": "BNK FEARX",
            "Winner": "",
            "Team1Score": "",
            "Team2Score": "",
            "DateTime UTC": "2026-03-16 13:00:00",
            "BestOf": "5",
            "OverviewPage": "First Stand Tournament/2026",
            "Tab": "Round 1",
            "MatchDay": "1",
            "Patch": "26.05",
        },
        {
            "MatchId": "Old_Match",
            "Team1": "A",
            "Team2": "B",
            "Winner": "",
            "Team1Score": "",
            "Team2Score": "",
            "DateTime UTC": "2026-03-14 13:00:00",
            "BestOf": "3",
            "OverviewPage": "Old Event 2026",
        },
        {
            "MatchId": "Resolved_Match",
            "Team1": "A",
            "Team2": "B",
            "Winner": "1",
            "Team1Score": "2",
            "Team2Score": "0",
            "DateTime UTC": "2026-03-17 13:00:00",
            "BestOf": "3",
            "OverviewPage": "Resolved Event 2026",
        },
        {
            "MatchId": "TBD_Match",
            "Team1": "TBD",
            "Team2": "B",
            "Winner": "",
            "Team1Score": "",
            "Team2Score": "",
            "DateTime UTC": "2026-03-17 15:00:00",
            "BestOf": "3",
            "OverviewPage": "Future Event 2026",
        },
    ]

    upcoming_df = normalize_upcoming_match_rows(rows, reference_time=pd.Timestamp("2026-03-15T00:00:00Z"))

    assert len(upcoming_df) == 1
    record = upcoming_df.iloc[0].to_dict()
    assert record["league"] == "First Stand Tournament"
    assert record["team1"] == "Bilibili Gaming"
    assert record["team2"] == "BNK FEARX"
    assert record["phase_label"] == "Round 1 · 1"
