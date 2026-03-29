from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from web.app import app


def test_team_route_renders_analytics_sections(monkeypatch):
    monkeypatch.setattr(
        "web.app.db.get_team_info",
        lambda name: {
            "teamname": "T1",
            "league": "LCK",
            "league_label": "LCK",
        },
    )
    monkeypatch.setattr(
        "web.app.db.get_team_roster",
        lambda name: pd.DataFrame(
            [
                {
                    "playername": "Faker",
                    "position": "mid",
                    "games": 12,
                    "winrate": 75.0,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_team_stats_by_split",
        lambda name: pd.DataFrame(
            [
                {
                    "year": 2026,
                    "league": "LCK",
                    "league_label": "LCK",
                    "split": "Spring",
                    "games": 18,
                    "wins": 13,
                    "winrate": 72.2,
                    "avg_length": 1915,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_team_titles",
        lambda name: pd.DataFrame(
            [
                {
                    "year": 2025,
                    "league": "LCK",
                    "league_label": "LCK",
                    "runner_up": "Gen.G",
                    "final_score": "3-2",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_team_recent_series",
        lambda name: pd.DataFrame(
            [
                {
                    "match_date": pd.Timestamp("2026-03-20"),
                    "league": "LCK",
                    "league_label": "LCK",
                    "team1": "T1",
                    "team2": "Gen.G",
                    "score": "3-1",
                    "series_winner": "T1",
                    "series_format": "Bo5",
                    "tournament_phase": "Final",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_team_betting_stats",
        lambda name: {
            "games": 24,
            "winrate": 66.7,
            "first_blood_pct": 58.3,
            "first_tower_pct": 62.5,
            "first_dragon_pct": 54.2,
            "first_herald_pct": 50.0,
            "first_baron_pct": 63.6,
            "avg_total_kills": 28.4,
            "avg_total_towers": 11.6,
            "avg_total_dragons": 4.9,
            "avg_total_nashors": 1.4,
            "avg_total_inhibitors": 2.2,
            "avg_game_minutes": 31.7,
            "kills_over_25_pct": 58.3,
            "towers_over_10_pct": 62.5,
            "nashors_over_1_5_pct": 29.2,
        },
    )
    monkeypatch.setattr(
        "web.app.db.get_team_winrate_by_split",
        lambda name: pd.DataFrame(
            [
                {
                    "year": 2025,
                    "split": "Summer",
                    "games": 18,
                    "winrate": 66.7,
                    "avg_minutes": 32.0,
                    "avg_kills": 15.4,
                },
                {
                    "year": 2026,
                    "split": "Spring",
                    "games": 18,
                    "winrate": 72.2,
                    "avg_minutes": 31.2,
                    "avg_kills": 16.1,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_team_form",
        lambda name, limit=10: pd.DataFrame(
            [
                {
                    "game_date": pd.Timestamp("2026-03-20"),
                    "result": 1,
                    "total_kills": 29,
                    "game_minutes": 32,
                    "opp_teamname": "Gen.G",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_team_analytics_summary",
        lambda name: {
            "games_recent": 20,
            "winrate_recent": 70.0,
            "winrate_delta": 9.5,
            "dominance_score": 76.2,
            "avg_gd15": 642.0,
            "objective_control_pct": 58.4,
            "blue_side_winrate": 75.0,
            "red_side_winrate": 64.3,
            "trend_label": "Heating up",
            "trend_note": "Recent win rate is materially above the previous window.",
            "early_game_label": "Fast starter",
            "early_game_note": "The team regularly builds leads before the game opens up.",
            "tempo_label": "Controlled pace",
            "tempo_note": "The team mixes proactive fights with structure-first setups.",
            "control_label": "Objective-led",
            "control_note": "Openers and structures are converting into reliable map control.",
            "side_bias_label": "Blue leaning",
            "side_bias_note": "Results are meaningfully better on blue side in the recent sample.",
        },
    )
    monkeypatch.setattr(
        "web.app.db.get_team_player_impact",
        lambda name: pd.DataFrame(
            [
                {
                    "playername": "Faker",
                    "position": "mid",
                    "games": 12,
                    "winrate": 75.0,
                    "avg_dpm": 645.0,
                    "avg_damage_share_pct": 28.4,
                    "avg_gold_share_pct": 24.8,
                    "avg_gd15": 521.0,
                    "avg_csd15": 8.2,
                    "avg_vision": 32.4,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_team_signature_champions",
        lambda name: pd.DataFrame(
            [
                {
                    "champion": "Ahri",
                    "position": "mid",
                    "games": 5,
                    "winrate": 80.0,
                    "avg_dpm": 670.0,
                    "avg_gd15": 430.0,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_team_patch_profile",
        lambda name: pd.DataFrame(
            [
                {
                    "patch": "26.05",
                    "games": 8,
                    "winrate": 75.0,
                    "avg_kill_diff": 4.8,
                    "avg_minutes": 31.1,
                    "first_dragon_pct": 62.5,
                }
            ]
        ),
    )

    client = TestClient(app)
    response = client.get("/team/T1")

    assert response.status_code == 200
    assert "Dominance Score" in response.text
    assert "Player Impact" in response.text
    assert "Signature Champions" in response.text
    assert "Patch Performance" in response.text
    assert "Heating up" in response.text
    assert 'data-series-team1="T1"' in response.text
