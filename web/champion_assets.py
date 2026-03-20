"""Helpers for champion artwork URLs used by the web UI."""

from __future__ import annotations

import copy
import json
import os
import re
from functools import lru_cache
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

DEFAULT_DDRAGON_VERSION = os.environ.get("RIOT_DDRAGON_VERSION", "16.6.1")
DDRAGON_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_SQUARE_URL = (
    "https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{champion_id}.png"
)
_REQUEST_TIMEOUT_SECONDS = 2.5

# Data Dragon ids are mostly display-name-based, with a few long-lived exceptions.
_IRREGULAR_CHAMPION_IDS = {
    "aurelionsol": "AurelionSol",
    "belveth": "Belveth",
    "chogath": "Chogath",
    "drmundo": "DrMundo",
    "jarvaniv": "JarvanIV",
    "kaisa": "KaiSa",
    "khazix": "Khazix",
    "kogmaw": "KogMaw",
    "ksante": "KSante",
    "leblanc": "Leblanc",
    "masteryi": "MasterYi",
    "missfortune": "MissFortune",
    "monkeyking": "MonkeyKing",
    "nunu": "Nunu",
    "nunuwillump": "Nunu",
    "reksai": "RekSai",
    "renata": "Renata",
    "renataglasc": "Renata",
    "tahmkench": "TahmKench",
    "twistedfate": "TwistedFate",
    "velkoz": "Velkoz",
    "wukong": "MonkeyKing",
    "xinzhao": "XinZhao",
}


def _normalized(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _title_cased_compound(value: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", value)
    if not parts:
        return ""

    out: list[str] = []
    for part in parts:
        if len(part) > 1 and any(char.isupper() for char in part[1:]):
            out.append(part)
        else:
            out.append(part[:1].upper() + part[1:].lower())
    return "".join(out)


@lru_cache(maxsize=1)
def get_datadragon_version() -> str:
    try:
        with urlopen(DDRAGON_VERSIONS_URL, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except (OSError, TimeoutError, URLError, ValueError):
        return DEFAULT_DDRAGON_VERSION

    if isinstance(payload, list) and payload:
        latest = payload[0]
        if isinstance(latest, str) and latest.strip():
            return latest.strip()
    return DEFAULT_DDRAGON_VERSION


def resolve_champion_id(champion_name: str | None) -> str | None:
    if champion_name is None:
        return None

    text = str(champion_name).strip()
    if not text:
        return None

    normalized = _normalized(text)
    if not normalized:
        return None

    if normalized in _IRREGULAR_CHAMPION_IDS:
        return _IRREGULAR_CHAMPION_IDS[normalized]

    return _title_cased_compound(text)


def get_champion_square_url(champion_name: str | None) -> str | None:
    champion_id = resolve_champion_id(champion_name)
    if not champion_id:
        return None

    return DDRAGON_SQUARE_URL.format(
        version=get_datadragon_version(),
        champion_id=champion_id,
    )


def enrich_series_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach champion image urls to the dynamic series-games payload."""
    enriched = copy.deepcopy(games)
    for game in enriched:
        for team_key in ("team1", "team2"):
            team = game.get(team_key)
            if not isinstance(team, dict):
                continue

            team["pick_cards"] = [
                {
                    "name": champion_name,
                    "image_url": get_champion_square_url(champion_name),
                }
                for champion_name in (team.get("picks") or [])
            ]
            team["ban_cards"] = [
                {
                    "name": champion_name,
                    "image_url": get_champion_square_url(champion_name),
                }
                for champion_name in (team.get("bans") or [])
            ]

            players = []
            for player in team.get("players") or []:
                player_record = dict(player)
                player_record["champion_image_url"] = get_champion_square_url(
                    player_record.get("champion")
                )
                players.append(player_record)
            team["players"] = players

    return enriched
