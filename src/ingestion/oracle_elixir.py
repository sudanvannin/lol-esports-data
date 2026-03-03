"""
Oracle's Elixir data downloader.

Oracle's Elixir provides aggregated pro play statistics in CSV format.
URL: https://oracleselixir.com/tools/downloads

Data available:
- Match-level stats (2014-present)
- Player-level stats per game
- Team-level stats per game

This serves as a backup/complement to the live API data.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

ORACLE_ELIXIR_BASE_URL = "https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com"

AVAILABLE_DATASETS = {
    "2024": f"{ORACLE_ELIXIR_BASE_URL}/2024_LoL_esports_match_data_from_OraclesElixir.csv",
    "2023": f"{ORACLE_ELIXIR_BASE_URL}/2023_LoL_esports_match_data_from_OraclesElixir.csv",
    "2022": f"{ORACLE_ELIXIR_BASE_URL}/2022_LoL_esports_match_data_from_OraclesElixir.csv",
    "2021": f"{ORACLE_ELIXIR_BASE_URL}/2021_LoL_esports_match_data_from_OraclesElixir.csv",
    "2020": f"{ORACLE_ELIXIR_BASE_URL}/2020_LoL_esports_match_data_from_OraclesElixir.csv",
    "2019": f"{ORACLE_ELIXIR_BASE_URL}/2019_LoL_esports_match_data_from_OraclesElixir.csv",
    "2018": f"{ORACLE_ELIXIR_BASE_URL}/2018_LoL_esports_match_data_from_OraclesElixir.csv",
    "2017": f"{ORACLE_ELIXIR_BASE_URL}/2017_LoL_esports_match_data_from_OraclesElixir.csv",
    "2016": f"{ORACLE_ELIXIR_BASE_URL}/2016_LoL_esports_match_data_from_OraclesElixir.csv",
    "2015": f"{ORACLE_ELIXIR_BASE_URL}/2015_LoL_esports_match_data_from_OraclesElixir.csv",
    "2014": f"{ORACLE_ELIXIR_BASE_URL}/2014_LoL_esports_match_data_from_OraclesElixir.csv",
}

RELEVANT_COLUMNS = [
    "gameid",
    "datacompleteness",
    "url",
    "league",
    "year",
    "split",
    "playoffs",
    "date",
    "game",
    "patch",
    "participantid",
    "side",
    "position",
    "playername",
    "playerid",
    "teamname",
    "teamid",
    "champion",
    "ban1",
    "ban2",
    "ban3",
    "ban4",
    "ban5",
    "gamelength",
    "result",
    "kills",
    "deaths",
    "assists",
    "teamkills",
    "teamdeaths",
    "doublekills",
    "triplekills",
    "quadrakills",
    "pentakills",
    "firstblood",
    "firstbloodkill",
    "firstbloodassist",
    "firstbloodvictim",
    "team kpm",
    "ckpm",
    "firstdragon",
    "dragons",
    "opp_dragons",
    "elementaldrakes",
    "opp_elementaldrakes",
    "infernals",
    "mountains",
    "clouds",
    "oceans",
    "chemtechs",
    "hextechs",
    "dragons (type unknown)",
    "elders",
    "opp_elders",
    "firstherald",
    "heralds",
    "opp_heralds",
    "firstbaron",
    "barons",
    "opp_barons",
    "firsttower",
    "towers",
    "opp_towers",
    "firstmidtower",
    "firsttothreetowers",
    "turretplates",
    "opp_turretplates",
    "inhibitors",
    "opp_inhibitors",
    "damagetochampions",
    "dpm",
    "damageshare",
    "damagetakenperminute",
    "damagemitigatedperminute",
    "wardsplaced",
    "wpm",
    "wardskilled",
    "wcpm",
    "controlwardsbought",
    "visionscore",
    "vspm",
    "totalgold",
    "earnedgold",
    "earned gpm",
    "earnedgoldshare",
    "goldspent",
    "gspd",
    "total cs",
    "minionkills",
    "monsterkills",
    "monsterkillsownjungle",
    "monsterkillsenemyjungle",
    "cspm",
    "goldat10",
    "xpat10",
    "csat10",
    "opp_goldat10",
    "opp_xpat10",
    "opp_csat10",
    "golddiffat10",
    "xpdiffat10",
    "csdiffat10",
    "killsat10",
    "assistsat10",
    "deathsat10",
    "opp_killsat10",
    "opp_assistsat10",
    "opp_deathsat10",
    "goldat15",
    "xpat15",
    "csat15",
    "opp_goldat15",
    "opp_xpat15",
    "opp_csat15",
    "golddiffat15",
    "xpdiffat15",
    "csdiffat15",
    "killsat15",
    "assistsat15",
    "deathsat15",
    "opp_killsat15",
    "opp_assistsat15",
    "opp_deathsat15",
]


@dataclass
class DownloadResult:
    """Result of a CSV download operation."""

    year: str
    success: bool
    rows: int
    file_path: str | None
    error: str | None = None


class OracleElixirDownloader:
    """Downloader for Oracle's Elixir CSV datasets."""

    def __init__(
        self,
        output_dir: str | Path = "data/oracle_elixir",
        timeout: float = 300.0,
    ):
        self.output_dir = Path(output_dir)
        self.timeout = timeout

    async def download_year(
        self,
        year: str,
        filter_leagues: list[str] | None = None,
    ) -> DownloadResult:
        """
        Download and optionally filter data for a specific year.

        Args:
            year: Year to download (e.g., "2024")
            filter_leagues: Optional list of league slugs to filter
                           (e.g., ["CBLOL", "LCK", "LEC"])

        Returns:
            DownloadResult with status and file path
        """
        if year not in AVAILABLE_DATASETS:
            return DownloadResult(
                year=year,
                success=False,
                rows=0,
                file_path=None,
                error=f"Year {year} not available. Valid years: {list(AVAILABLE_DATASETS.keys())}",
            )

        url = AVAILABLE_DATASETS[year]
        logger.info(f"Downloading {year} data from Oracle's Elixir...")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()

            df = pd.read_csv(
                pd.io.common.StringIO(response.text),
                low_memory=False,
            )

            logger.info(f"Downloaded {len(df)} rows for {year}")

            if filter_leagues:
                original_count = len(df)
                df = df[df["league"].str.upper().isin([l.upper() for l in filter_leagues])]
                logger.info(
                    f"Filtered to {len(df)} rows (from {original_count}) for leagues: {filter_leagues}"
                )

            self.output_dir.mkdir(parents=True, exist_ok=True)

            if filter_leagues:
                leagues_str = "_".join(sorted(filter_leagues))
                filename = f"{year}_{leagues_str}_oracle_elixir.csv"
            else:
                filename = f"{year}_oracle_elixir.csv"

            file_path = self.output_dir / filename
            df.to_csv(file_path, index=False)

            return DownloadResult(
                year=year,
                success=True,
                rows=len(df),
                file_path=str(file_path),
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error downloading {year}: {e.response.status_code}"
            logger.error(error_msg)
            return DownloadResult(
                year=year,
                success=False,
                rows=0,
                file_path=None,
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Error downloading {year}: {str(e)}"
            logger.error(error_msg)
            return DownloadResult(
                year=year,
                success=False,
                rows=0,
                file_path=None,
                error=error_msg,
            )

    async def download_all(
        self,
        years: list[str] | None = None,
        filter_leagues: list[str] | None = None,
    ) -> list[DownloadResult]:
        """
        Download multiple years of data.

        Args:
            years: List of years to download (defaults to all available)
            filter_leagues: Optional list of leagues to filter

        Returns:
            List of DownloadResults
        """
        if years is None:
            years = list(AVAILABLE_DATASETS.keys())

        tasks = [self.download_year(year, filter_leagues) for year in years]
        results = await asyncio.gather(*tasks)

        return list(results)

    def load_csv(self, year: str) -> pd.DataFrame | None:
        """Load a previously downloaded CSV."""
        pattern = f"{year}*_oracle_elixir.csv"
        files = list(self.output_dir.glob(pattern))

        if not files:
            logger.warning(f"No CSV found for {year}")
            return None

        file_path = files[0]
        logger.info(f"Loading {file_path}")

        return pd.read_csv(file_path, low_memory=False)


def parse_oracle_data(df: pd.DataFrame) -> dict[str, Any]:
    """
    Parse Oracle's Elixir data into structured format.

    The raw data has one row per player per game (10 rows per game).
    This function reorganizes it into game-level and team-level structures.
    """
    games = {}

    for game_id, game_df in df.groupby("gameid"):
        team_rows = game_df[game_df["position"] == "team"]
        player_rows = game_df[game_df["position"] != "team"]

        if len(team_rows) != 2:
            continue

        blue_team = team_rows[team_rows["side"] == "Blue"].iloc[0] if len(team_rows[team_rows["side"] == "Blue"]) > 0 else None
        red_team = team_rows[team_rows["side"] == "Red"].iloc[0] if len(team_rows[team_rows["side"] == "Red"]) > 0 else None

        if blue_team is None or red_team is None:
            continue

        blue_players = player_rows[player_rows["side"] == "Blue"].to_dict("records")
        red_players = player_rows[player_rows["side"] == "Red"].to_dict("records")

        games[game_id] = {
            "game_id": game_id,
            "league": blue_team.get("league"),
            "year": blue_team.get("year"),
            "split": blue_team.get("split"),
            "playoffs": blue_team.get("playoffs"),
            "date": blue_team.get("date"),
            "patch": blue_team.get("patch"),
            "game_length": blue_team.get("gamelength"),
            "blue_team": {
                "name": blue_team.get("teamname"),
                "id": blue_team.get("teamid"),
                "result": blue_team.get("result"),
                "kills": blue_team.get("teamkills"),
                "deaths": blue_team.get("teamdeaths"),
                "dragons": blue_team.get("dragons"),
                "barons": blue_team.get("barons"),
                "towers": blue_team.get("towers"),
                "first_blood": blue_team.get("firstblood"),
                "first_dragon": blue_team.get("firstdragon"),
                "first_baron": blue_team.get("firstbaron"),
                "first_tower": blue_team.get("firsttower"),
                "gold_at_15": blue_team.get("goldat15"),
                "players": blue_players,
            },
            "red_team": {
                "name": red_team.get("teamname"),
                "id": red_team.get("teamid"),
                "result": red_team.get("result"),
                "kills": red_team.get("teamkills"),
                "deaths": red_team.get("teamdeaths"),
                "dragons": red_team.get("dragons"),
                "barons": red_team.get("barons"),
                "towers": red_team.get("towers"),
                "first_blood": red_team.get("firstblood"),
                "first_dragon": red_team.get("firstdragon"),
                "first_baron": red_team.get("firstbaron"),
                "first_tower": red_team.get("firsttower"),
                "gold_at_15": red_team.get("goldat15"),
                "players": red_players,
            },
        }

    return games


async def main():
    """Example usage."""
    downloader = OracleElixirDownloader(output_dir="data/oracle_elixir")

    result = await downloader.download_year(
        year="2024",
        filter_leagues=["CBLOL", "LCK", "LEC", "LCS"],
    )

    if result.success:
        print(f"Downloaded {result.rows} rows to {result.file_path}")

        df = downloader.load_csv("2024")
        if df is not None:
            print(f"\nDataset shape: {df.shape}")
            print(f"Leagues: {df['league'].unique()}")
            print(f"Games: {df['gameid'].nunique()}")
    else:
        print(f"Error: {result.error}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
