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
        "web.app.prob_win.build_match_explainability",
        lambda match_row: {
            "available": True,
            "favorite_name": "Bilibili Gaming",
            "favorite_prob_pct": 77.3,
            "confidence_gap_pct": 54.6,
            "confidence_label": "High conviction",
            "confidence_note": "The model gap is wide and the underlying ensemble is tightly aligned.",
            "summary": "Bilibili Gaming projects as the favorite mainly through recent form and patch fit.",
            "signals": [
                {
                    "title": "Recent form",
                    "favored_name": "Bilibili Gaming",
                    "strength_label": "Strong",
                    "value_text": "18.0pp",
                    "note": "Last-five-series win-rate gap inside the core matchup history.",
                }
            ],
            "sample_context": [
                {
                    "label": "Core series",
                    "team1_value": 12,
                    "team2_value": 11,
                }
            ],
            "model_agreement": {
                "label": "Tight agreement",
                "note": "The ensemble models are tightly clustered around the same read.",
                "disagreement_pct": 2.4,
                "calibration_method": "isotonic",
                "models": [
                    {
                        "model_name": "Xgboost",
                        "team1_win_pct": 76.9,
                    }
                ],
            },
            "warnings": [],
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
    assert "Explainability" in response.text
    assert "High conviction" in response.text
    assert "Recent form" in response.text
    assert "Total Kills" in response.text


def test_home_route_renders_versioned_css_and_series_dataset_attrs(monkeypatch):
    monkeypatch.setattr(
        "web.app.db.get_home_recent_series",
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
        "web.app.db.get_home_analytics_summary",
        lambda: {
            "completed_series_7d": 12,
            "completed_series_30d": 48,
            "active_leagues_30d": 4,
            "tracked_games_30d": 96,
            "tracked_teams_90d": 20,
            "avg_game_minutes_30d": 31.4,
            "avg_total_kills_30d": 27.8,
            "upcoming_matches": 7,
            "recommended_edges": 2,
        },
    )
    monkeypatch.setattr(
        "web.app.db.get_power_rankings",
        lambda limit=10: pd.DataFrame(
            [
                {
                    "rank": 1,
                    "teamname": "Bilibili Gaming",
                    "league": "FST",
                    "league_label": "First Stand",
                    "games": 12,
                    "wins": 9,
                    "winrate": 75.0,
                    "last5_games": 5,
                    "last5_wins": 4,
                    "last5_winrate": 80.0,
                    "avg_kill_diff": 4.2,
                    "avg_tower_diff": 1.8,
                    "avg_dragon_diff": 1.1,
                    "first_blood_pct": 66.7,
                    "first_tower_pct": 58.3,
                    "power_score": 83.4,
                    "trend_label": "Surging",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_league_trends",
        lambda limit=6: pd.DataFrame(
            [
                {
                    "league": "FST",
                    "league_label": "First Stand",
                    "games": 18,
                    "teams": 8,
                    "avg_total_kills": 29.4,
                    "avg_game_minutes": 32.1,
                    "kill_pace": 0.92,
                    "tempo_label": "Measured",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_home_model_watchlist",
        lambda limit=6: pd.DataFrame(
            [
                {
                    "match_id": "fst_match_1",
                    "match_time": pd.Timestamp("2026-03-18T13:00:00Z"),
                    "league": "FST",
                    "league_label": "First Stand",
                    "team1": "Bilibili Gaming",
                    "team2": "BNK FEARX",
                    "favorite_name": "Bilibili Gaming",
                    "favorite_prob_pct": 72.5,
                    "confidence_gap_pct": 45.0,
                    "team1_fair_odds": 1.38,
                    "team2_fair_odds": 3.55,
                    "angle_label": "Strong favorite",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_home_edge_highlights",
        lambda limit=5: pd.DataFrame(
            [
                {
                    "captured_at": "2026-03-17T12:00:00Z",
                    "match_time": "2026-03-18T13:00:00Z",
                    "league_code": "FST",
                    "league_code_label": "First Stand",
                    "team1_name": "Bilibili Gaming",
                    "team2_name": "BNK FEARX",
                    "bookmaker": "Bet365",
                    "recommended_selection": "Bilibili Gaming",
                    "recommended_odds": 1.91,
                    "recommended_fair_odds": 1.72,
                    "recommended_edge_pct": 5.4,
                    "recommended_ev_pct": 9.1,
                    "recommended_kelly_pct": 8.2,
                    "recommendation_reason": "Highest EV side above thresholds.",
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
    assert "Power Rankings" in response.text
    assert "Model Spotlight" in response.text
    assert 'data-series-team1="Bilibili Gaming"' in response.text
    assert 'data-series-phase="Semifinals"' in response.text


def test_edge_board_route_renders(monkeypatch):
    monkeypatch.setattr(
        "web.app.db.get_edge_board",
        lambda limit=80, league=None, bookmaker=None, recommended_only=True: pd.DataFrame(
            [
                {
                    "captured_at": "2026-03-23T12:00:00Z",
                    "match_time": "2026-03-24T15:00:00Z",
                    "league_code": "FST",
                    "league_code_label": "First Stand",
                    "team1_name": "Bilibili Gaming",
                    "team2_name": "BNK FEARX",
                    "bookmaker": "Bet365",
                    "recommend_bet": True,
                    "recommendation_reason": "Highest EV side above thresholds.",
                    "recommended_selection": "Bilibili Gaming",
                    "recommended_odds": 1.91,
                    "recommended_fair_odds": 1.75,
                    "recommended_model_probability": 0.57,
                    "recommended_edge_pct": 4.6,
                    "recommended_ev_pct": 8.9,
                    "recommended_kelly_pct": 9.8,
                    "model_run_id": "robust_run_1",
                    "snapshot_id": "snapshot_1",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_upcoming_match_leagues",
        lambda: pd.DataFrame(
            [
                {
                    "league": "FST",
                    "league_label": "First Stand",
                    "total_matches": 1,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_edge_board_bookmakers",
        lambda: pd.DataFrame(
            [
                {
                    "bookmaker": "Bet365",
                    "total_rows": 1,
                    "latest_captured_at": "2026-03-23T12:00:00Z",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_edge_board_meta",
        lambda: {
            "row_count": 1,
            "recommendation_count": 1,
            "bookmaker_count": 1,
            "latest_captured_at": "2026-03-23T12:00:00Z",
        },
    )

    client = TestClient(app)
    response = client.get("/edge-board")

    assert response.status_code == 200
    assert "Edge Board" in response.text
    assert "Bilibili Gaming vs BNK FEARX" in response.text
    assert "Recommend" in response.text
