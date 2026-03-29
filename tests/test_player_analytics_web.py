from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from web.app import app


def test_player_route_renders_analytics_sections(monkeypatch):
    monkeypatch.setattr(
        "web.app.db.get_player_info",
        lambda name: {
            "playername": "Faker",
            "teamname": "T1",
            "position": "mid",
            "league": "LCK",
            "league_label": "LCK",
        },
    )
    monkeypatch.setattr(
        "web.app.db.get_player_career_stats",
        lambda name, year=None, split=None: {
            "games": 22,
            "wins": 15,
            "winrate": 68.2,
            "avg_kills": 4.5,
            "avg_deaths": 2.1,
            "avg_assists": 7.8,
            "kda": 5.86,
            "avg_dpm": 645.0,
            "avg_cspm": 8.8,
            "avg_gd15": 322.0,
            "avg_xd15": 210.0,
            "avg_vs": 30.2,
        },
    )
    monkeypatch.setattr(
        "web.app.db.get_player_by_year",
        lambda name: pd.DataFrame(
            [
                {
                    "year": 2025,
                    "split": "Summer",
                    "teamname": "T1",
                    "games": 18,
                    "wins": 12,
                    "winrate": 66.7,
                    "avg_k": 4.1,
                    "avg_d": 2.3,
                    "avg_a": 7.0,
                    "avg_dpm": 610.0,
                },
                {
                    "year": 2026,
                    "split": "Spring",
                    "teamname": "T1",
                    "games": 22,
                    "wins": 15,
                    "winrate": 68.2,
                    "avg_k": 4.5,
                    "avg_d": 2.1,
                    "avg_a": 7.8,
                    "avg_dpm": 645.0,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_player_recent_games",
        lambda name, year=None, split=None: pd.DataFrame(
            [
                {
                    "game_date": pd.Timestamp("2026-03-20"),
                    "league": "LCK",
                    "league_label": "LCK",
                    "split": "Spring",
                    "teamname": "T1",
                    "champion": "Ahri",
                    "position": "mid",
                    "kills": 7,
                    "deaths": 2,
                    "assists": 9,
                    "dpm": 702,
                    "cspm": 8.9,
                    "result": 1,
                },
                {
                    "game_date": pd.Timestamp("2026-03-18"),
                    "league": "LCK",
                    "league_label": "LCK",
                    "split": "Spring",
                    "teamname": "T1",
                    "champion": "Orianna",
                    "position": "mid",
                    "kills": 4,
                    "deaths": 1,
                    "assists": 11,
                    "dpm": 618,
                    "cspm": 8.7,
                    "result": 1,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_player_splits",
        lambda name: pd.DataFrame(
            [
                {"year": 2026, "split": "Spring"},
                {"year": 2025, "split": "Summer"},
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_player_analytics_summary",
        lambda name, year=None, split=None: {
            "available": True,
            "games": 22,
            "wins": 15,
            "winrate": 68.2,
            "recent_games": 12,
            "recent_winrate": 75.0,
            "previous_winrate": 58.3,
            "winrate_delta": 16.7,
            "avg_kda": 5.86,
            "avg_dpm": 645.0,
            "avg_cspm": 8.8,
            "avg_gd15": 322.0,
            "avg_xd15": 210.0,
            "avg_csd15": 7.1,
            "avg_damage_share_pct": 28.4,
            "avg_gold_share_pct": 24.1,
            "avg_earned_gpm": 432.0,
            "avg_vision": 30.2,
            "avg_vspm": 1.42,
            "first_blood_involvement_pct": 31.8,
            "unique_champions": 8,
            "top_champion_share_pct": 22.7,
            "top3_champion_share_pct": 54.5,
            "impact_index": 84.5,
            "trend_label": "Heating up",
            "trend_note": "Recent output is materially ahead of the previous window.",
            "lane_label": "Lane driver",
            "lane_note": "The player tends to create measurable lane advantages before 15 minutes.",
            "role_label": "Primary carry",
            "role_note": "The player is taking a large share of team resources and converting them into output.",
            "setup_label": "Early instigator",
            "setup_note": "The player is regularly present in first-blood sequences.",
            "pool_label": "Flexible core",
            "pool_note": "There is a stable comfort set without becoming overly narrow.",
        },
    )
    monkeypatch.setattr(
        "web.app.db.get_player_role_benchmark",
        lambda name, year=None, split=None: {
            "available": True,
            "position": "mid",
            "peer_count": 24,
            "scope_label": "2026 Spring",
            "impact_index": 84.5,
            "impact_label": "Top quartile role impact",
            "impact_note": "This player rates near the top of the position on the current analytical mix.",
            "metrics": [
                {
                    "metric": "DPM",
                    "player_value": 645.0,
                    "peer_avg": 571.0,
                    "delta_text": "+74",
                    "percentile_label": "88th pct",
                    "strength_label": "Elite",
                    "tone": "up",
                }
            ],
        },
    )
    monkeypatch.setattr(
        "web.app.db.get_player_champion_analytics",
        lambda name, year=None, split=None: pd.DataFrame(
            [
                {
                    "champion": "Ahri",
                    "games": 5,
                    "pick_share_pct": 22.7,
                    "winrate": 80.0,
                    "avg_k": 5.4,
                    "avg_d": 1.8,
                    "avg_a": 8.0,
                    "avg_dpm": 688.0,
                    "avg_gd15": 290.0,
                    "avg_damage_share_pct": 29.1,
                    "profile_label": "Carry pick",
                }
            ]
        ),
    )

    client = TestClient(app)
    response = client.get("/player/Faker?year=2026&split=Spring")

    assert response.status_code == 200
    assert "Role Benchmark" in response.text
    assert "Champion Signatures" in response.text
    assert "Top quartile role impact" in response.text
    assert "Lane driver" in response.text
    assert "Ahri" in response.text
