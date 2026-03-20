from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from web import prob_win
from web.app import app


def test_build_match_context_maps_first_stand():
    context = prob_win.build_match_context(
        {
            "match_id": "fst_match_1",
            "match_time": pd.Timestamp("2026-03-16T13:00:00Z"),
            "league": "First Stand",
            "event_name": "2026 First Stand",
            "phase_label": "Groups Day 2",
            "team1": "Bilibili Gaming",
            "team2": "BNK FEARX",
            "best_of": 5,
            "patch": "26.05",
        }
    )

    assert context["league_code"] == "FST"
    assert context["split_name"] == "FST"
    assert context["best_of"] == 5
    assert context["patch_version"] == "26.05"


def test_prob_win_route_renders(monkeypatch):
    rows_df = pd.DataFrame(
        [
            {
                "match_id": "fst_match_1",
                "match_time": pd.Timestamp("2026-03-16T13:00:00Z"),
                "match_date": pd.Timestamp("2026-03-16T00:00:00Z"),
                "league": "First Stand",
                "league_label": "First Stand",
                "event_name": "2026 First Stand",
                "phase_label": "Groups Day 2",
                "team1": "Bilibili Gaming",
                "team2": "BNK FEARX",
                "best_of": 5,
                "patch": "26.05",
                "source": "riot_official_schedule",
            }
        ]
    )
    leagues_df = pd.DataFrame(
        [
            {
                "league": "First Stand",
                "league_label": "First Stand",
                "total_matches": 1,
            }
        ]
    )

    monkeypatch.setattr(
        "web.app.db.get_upcoming_matches", lambda limit=40, league=None: rows_df
    )
    monkeypatch.setattr(
        "web.app.db.get_upcoming_match", lambda match_id: rows_df.iloc[0].to_dict()
    )
    monkeypatch.setattr("web.app.db.get_upcoming_match_leagues", lambda: leagues_df)
    monkeypatch.setattr(
        "web.app.db.get_prob_win_matches",
        lambda match_ids=None: pd.DataFrame(
            [
                {
                    "match_id": "fst_match_1",
                    "winner_available": True,
                    "winner_error": "",
                    "team1_win_prob": 0.773,
                    "team2_win_prob": 0.227,
                    "team1_win_pct": 77.3,
                    "team2_win_pct": 22.7,
                    "team1_fair_odds": 1.29,
                    "team2_fair_odds": 4.41,
                    "favorite_name": "Bilibili Gaming",
                    "favorite_prob_pct": 77.3,
                    "warnings_json": "[]",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_prob_win_detail",
        lambda match_id: {
            "match": rows_df.iloc[0].to_dict(),
            "context": {
                "league_code": "FST",
                "split_name": "FST",
                "patch_version": "26.05",
                "best_of": 5,
                "playoffs": False,
            },
            "winner_market": {
                "available": True,
                "team1_name": "Bilibili Gaming",
                "team2_name": "BNK FEARX",
                "team1_win_pct": 77.3,
                "team2_win_pct": 22.7,
                "team1_fair_odds": 1.29,
                "team2_fair_odds": 4.41,
            },
            "totals_market": {
                "available": True,
                "markets": [
                    {
                        "market_label": "Total Kills",
                        "line": 29.5,
                        "distribution": "negative binomial",
                        "over_prob_pct": 58.7,
                        "over_fair_odds": 1.7,
                        "under_prob_pct": 41.3,
                        "under_fair_odds": 2.42,
                        "predicted_mean": 31.12,
                    }
                ],
            },
            "warnings": [],
            "has_any_market": True,
            "confidence_gap_pct": 54.6,
        },
    )
    monkeypatch.setattr(
        "web.app.db.get_upcoming_matches_meta",
        lambda: {
            "source": "riot_official_schedule",
            "fetched_at": "2026-03-16T22:00:00Z",
            "row_count": 1,
        },
    )

    client = TestClient(app)
    response = client.get("/prob-win?match_id=fst_match_1")

    assert response.status_code == 200
    assert "Prob Win" in response.text
    assert "Bilibili Gaming vs BNK FEARX" in response.text
    assert "Fair match odds" in response.text
    assert "Total Kills" in response.text


def test_home_route_renders_versioned_css_and_series_dataset_attrs(monkeypatch):
    monkeypatch.setattr(
        "web.app.db.get_recent_series",
        lambda limit=25: pd.DataFrame(
            [
                {
                    "match_date": pd.Timestamp("2026-03-16T00:00:00Z"),
                    "league": "FST",
                    "league_label": "First Stand",
                    "team1": "Bilibili Gaming",
                    "team2": "BNK FEARX",
                    "score": "3-1",
                    "series_winner": "Bilibili Gaming",
                    "series_format": "Bo5",
                    "tournament_phase": "Semifinals",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_active_leagues",
        lambda: pd.DataFrame(
            [
                {
                    "league": "FST",
                    "league_label": "First Stand",
                    "total_series": 8,
                    "last_match": pd.Timestamp("2026-03-16T00:00:00Z"),
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_upcoming_matches",
        lambda limit=8: pd.DataFrame(
            [
                {
                    "match_id": "fst_match_1",
                    "match_time": pd.Timestamp("2026-03-18T13:00:00Z"),
                    "league": "FST",
                    "league_label": "First Stand",
                    "event_name": "First Stand 2026",
                    "phase_label": "Semifinals",
                    "team1": "Bilibili Gaming",
                    "team2": "BNK FEARX",
                    "best_of": 5,
                    "patch": "26.05",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_upcoming_matches_meta",
        lambda: {"fetched_at": "2026-03-16T22:00:00Z"},
    )

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "/static/style.css?v=" in response.text
    assert 'data-series-team1="Bilibili Gaming"' in response.text
    assert 'data-series-phase="Semifinals"' in response.text
