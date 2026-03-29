from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from web.app import app


def test_game_route_renders_analytics_sections(monkeypatch):
    monkeypatch.setattr(
        "web.app.db.get_game_players",
        lambda gameid: pd.DataFrame(
            [
                {
                    "teamname": "T1",
                    "playername": "Zeus",
                    "position": "top",
                    "champion": "Gnar",
                    "side": "Blue",
                    "result": 1,
                    "kills": 4,
                    "deaths": 1,
                    "assists": 7,
                    "dpm": 540,
                    "damageshare": 0.22,
                    "totalgold": 13200,
                    "cspm": 8.4,
                    "visionscore": 24,
                    "golddiffat15": 380,
                    "xpdiffat15": 260,
                    "csdiffat15": 11,
                },
                {
                    "teamname": "Gen.G",
                    "playername": "Kiin",
                    "position": "top",
                    "champion": "K'Sante",
                    "side": "Red",
                    "result": 0,
                    "kills": 1,
                    "deaths": 4,
                    "assists": 2,
                    "dpm": 390,
                    "damageshare": 0.16,
                    "totalgold": 11200,
                    "cspm": 7.6,
                    "visionscore": 19,
                    "golddiffat15": -380,
                    "xpdiffat15": -260,
                    "csdiffat15": -11,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_game_teams",
        lambda gameid: pd.DataFrame(
            [
                {
                    "game_date": pd.Timestamp("2026-03-21"),
                    "league": "LCK",
                    "league_label": "LCK",
                    "patch": "26.05",
                    "game": 3,
                    "teamname": "T1",
                    "side": "Blue",
                    "result": 1,
                    "gamelength": 1920,
                    "teamkills": 17,
                    "teamdeaths": 8,
                    "pick1": "Gnar",
                    "pick2": "Vi",
                    "pick3": "Ahri",
                    "pick4": "Kai'Sa",
                    "pick5": "Rell",
                    "ban1": "Kalista",
                    "ban2": "Corki",
                    "ban3": "Skarner",
                    "ban4": "Taliyah",
                    "ban5": "Nautilus",
                    "towers": 9,
                    "dragons": 3,
                    "barons": 1,
                    "heralds": 1,
                    "elders": 0,
                    "void_grubs": 4,
                    "firstblood": 1,
                    "firsttower": 1,
                    "firstdragon": 1,
                    "firstbaron": 1,
                },
                {
                    "game_date": pd.Timestamp("2026-03-21"),
                    "league": "LCK",
                    "league_label": "LCK",
                    "patch": "26.05",
                    "game": 3,
                    "teamname": "Gen.G",
                    "side": "Red",
                    "result": 0,
                    "gamelength": 1920,
                    "teamkills": 8,
                    "teamdeaths": 17,
                    "pick1": "K'Sante",
                    "pick2": "Sejuani",
                    "pick3": "Azir",
                    "pick4": "Varus",
                    "pick5": "Alistar",
                    "ban1": "Yone",
                    "ban2": "Smolder",
                    "ban3": "Ashe",
                    "ban4": "Renata",
                    "ban5": "Poppy",
                    "towers": 3,
                    "dragons": 1,
                    "barons": 0,
                    "heralds": 0,
                    "elders": 0,
                    "void_grubs": 2,
                    "firstblood": 0,
                    "firsttower": 0,
                    "firstdragon": 0,
                    "firstbaron": 0,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_game_analytics_summary",
        lambda gameid: {
            "league": "LCK",
            "league_label": "LCK",
            "game_date": pd.Timestamp("2026-03-21"),
            "patch": "26.05",
            "game_number": 3,
            "winning_team": "T1",
            "game_minutes": 32.0,
            "script_label": "Clean snowball",
            "script_note": "T1 converted lane pressure into structures and never let the map reset.",
            "decisive_edge_label": "Early game lead",
            "decisive_edge_note": "T1 built roughly 1400 team GD@15.",
            "backdrop_label": "Result matched form",
            "backdrop_note": "T1 entered with the stronger recent profile and played to that level.",
            "h2h_total_series": 6,
            "winner_h2h_wins": 4,
            "loser_h2h_wins": 2,
            "h2h_note": "Tracked series history also leans T1 4-2.",
            "objective_share_pct": 72.0,
            "kill_diff": 9,
            "tower_diff": 6,
            "winner_gd15": 1400,
        },
    )
    monkeypatch.setattr(
        "web.app.db.get_game_team_backdrop",
        lambda gameid: pd.DataFrame(
            [
                {
                    "teamname": "T1",
                    "side": "Blue",
                    "won_game": True,
                    "games_recent": 18,
                    "winrate_recent": 72.2,
                    "winrate_delta": 8.5,
                    "dominance_score": 81.4,
                    "avg_gd15": 640,
                    "objective_control_pct": 61.2,
                    "trend_label": "Heating up",
                    "trend_note": "Recent win rate is materially above the previous window.",
                    "tempo_label": "Controlled pace",
                    "tempo_note": "The team mixes proactive fights with structure-first setups.",
                    "control_label": "Objective-led",
                    "control_note": "Openers and structures are converting into reliable map control.",
                },
                {
                    "teamname": "Gen.G",
                    "side": "Red",
                    "won_game": False,
                    "games_recent": 18,
                    "winrate_recent": 61.1,
                    "winrate_delta": -4.2,
                    "dominance_score": 69.7,
                    "avg_gd15": 210,
                    "objective_control_pct": 53.4,
                    "trend_label": "Stable",
                    "trend_note": "Results are broadly in line with the prior window.",
                    "tempo_label": "Measured tempo",
                    "tempo_note": "The team plays lower-event games and leans on cleaner setups.",
                    "control_label": "Contest-ready",
                    "control_note": "The team stays competitive across openings without one extreme identity.",
                },
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_game_lane_matchups",
        lambda gameid: pd.DataFrame(
            [
                {
                    "position": "top",
                    "position_label": "Top",
                    "team1_name": "T1",
                    "team1_player": "Zeus",
                    "team1_champion": "Gnar",
                    "team1_kda": 11.0,
                    "team1_dpm": 540,
                    "team1_damage_share_pct": 22.0,
                    "team1_vision": 24.0,
                    "team1_gd15": 380,
                    "team2_name": "Gen.G",
                    "team2_player": "Kiin",
                    "team2_champion": "K'Sante",
                    "team2_kda": 0.75,
                    "team2_dpm": 390,
                    "team2_damage_share_pct": 16.0,
                    "team2_vision": 19.0,
                    "team2_gd15": -380,
                    "edge_team": "T1",
                    "edge_label": "Clear edge",
                    "edge_note": "T1 created the clearest advantage through the lane phase.",
                }
            ]
        ),
    )

    client = TestClient(app)
    response = client.get("/game/lck_2026_t1_geng_g3")

    assert response.status_code == 200
    assert "Matchup Backdrop" in response.text
    assert "Lane Matchups" in response.text
    assert "Clean snowball" in response.text
    assert "Series History" in response.text
    assert "Zeus" in response.text
