from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from web import champion_assets
from web.app import app


def test_resolve_champion_id_handles_irregular_names():
    assert champion_assets.resolve_champion_id("Wukong") == "MonkeyKing"
    assert champion_assets.resolve_champion_id("Nunu & Willump") == "Nunu"
    assert champion_assets.resolve_champion_id("Renata Glasc") == "Renata"
    assert champion_assets.resolve_champion_id("Kai'Sa") == "KaiSa"
    assert champion_assets.resolve_champion_id("Aurelion Sol") == "AurelionSol"


def test_get_champion_square_url_uses_cached_version(monkeypatch):
    monkeypatch.setattr(champion_assets, "get_datadragon_version", lambda: "16.6.1")

    url = champion_assets.get_champion_square_url("Cho'Gath")

    assert url == "https://ddragon.leagueoflegends.com/cdn/16.6.1/img/champion/Chogath.png"


def test_enrich_series_games_adds_champion_urls(monkeypatch):
    monkeypatch.setattr(
        champion_assets,
        "get_champion_square_url",
        lambda name: f"https://cdn.test/{name.replace(' ', '_')}.png" if name else None,
    )

    payload = [
        {
            "gameid": "1",
            "team1": {
                "name": "Blue Team",
                "picks": ["Ahri", "Lee Sin"],
                "bans": ["LeBlanc"],
                "players": [
                    {
                        "player": "Faker",
                        "champion": "Ahri",
                    }
                ],
            },
            "team2": {
                "name": "Red Team",
                "picks": ["Renata Glasc"],
                "bans": [],
                "players": [
                    {
                        "player": "Chovy",
                        "champion": "Renata Glasc",
                    }
                ],
            },
        }
    ]

    enriched = champion_assets.enrich_series_games(payload)

    assert enriched[0]["team1"]["pick_cards"][0]["image_url"] == "https://cdn.test/Ahri.png"
    assert (
        enriched[0]["team1"]["ban_cards"][0]["image_url"]
        == "https://cdn.test/LeBlanc.png"
    )
    assert (
        enriched[0]["team2"]["players"][0]["champion_image_url"]
        == "https://cdn.test/Renata_Glasc.png"
    )
    assert "pick_cards" not in payload[0]["team1"]


def test_player_route_renders_champion_avatar(monkeypatch):
    monkeypatch.setattr(champion_assets, "get_datadragon_version", lambda: "16.6.1")
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
            "games": 10,
            "winrate": 70.0,
            "kda": 4.2,
            "avg_kills": 4.0,
            "avg_deaths": 2.0,
            "avg_assists": 6.0,
            "avg_dpm": 610.0,
            "avg_cspm": 8.7,
        },
    )
    monkeypatch.setattr(
        "web.app.db.get_player_by_year",
        lambda name: pd.DataFrame(
            [
                {
                    "year": 2026,
                    "split": "Spring",
                    "teamname": "T1",
                    "games": 10,
                    "winrate": 70.0,
                    "avg_k": 4.0,
                    "avg_d": 2.0,
                    "avg_a": 6.0,
                    "avg_dpm": 610.0,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_player_champions",
        lambda name, year=None, split=None: pd.DataFrame(
            [
                {
                    "champion": "Ahri",
                    "games": 4,
                    "winrate": 75.0,
                    "avg_k": 5.0,
                    "avg_d": 2.0,
                    "avg_a": 7.0,
                    "avg_dpm": 640.0,
                }
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
                    "champion": "Ahri",
                    "kills": 5,
                    "deaths": 1,
                    "assists": 8,
                    "dpm": 650.0,
                    "cspm": 8.9,
                    "result": 1,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "web.app.db.get_player_splits",
        lambda name: pd.DataFrame([{"year": 2026, "split": "Spring"}]),
    )

    client = TestClient(app)
    response = client.get("/player/Faker")

    assert response.status_code == 200
    assert "Ahri" in response.text
    assert (
        "https://ddragon.leagueoflegends.com/cdn/16.6.1/img/champion/Ahri.png"
        in response.text
    )
