"""
Data collectors that orchestrate ingestion to Bronze layer.

Collectors fetch data from sources and save to MinIO/S3 in the Bronze layer
following the Medallion architecture.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .esports_api import LoLEsportsClient, LoLEsportsFeedClient
from .oracle_elixir import OracleElixirDownloader

logger = logging.getLogger(__name__)


@dataclass
class CollectorConfig:
    """Configuration for data collectors."""

    local_backup_dir: str = "data/bronze"


class BronzeStorage:
    """Handles storage of raw data in Bronze layer."""

    def __init__(self, config: CollectorConfig):
        self.config = config
        self.local_dir = Path(config.local_backup_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

    def save_json(
        self,
        data: dict | list,
        data_type: str,
        identifier: str,
        timestamp: datetime | None = None,
    ) -> str:
        """
        Save JSON data to Bronze layer.

        Args:
            data: Data to save
            data_type: Type of data (leagues, tournaments, matches, games)
            identifier: Unique identifier for the file
            timestamp: Timestamp for partitioning (defaults to now)

        Returns:
            Path/key where data was saved
        """
        timestamp = timestamp or datetime.utcnow()

        json_str = json.dumps(data, default=str, indent=2)

        local_path = self.local_dir / data_type / timestamp.strftime("%Y/%m/%d")
        local_path.mkdir(parents=True, exist_ok=True)
        local_file = local_path / f"{identifier}.json"
        local_file.write_text(json_str)

        logger.debug(f"Saved locally: {local_file}")

        return str(local_file)

    def save_csv(
        self,
        file_path: str | Path,
        data_type: str,
        identifier: str,
    ) -> str:
        """Upload a CSV file to Bronze layer."""
        file_path = Path(file_path)
        
        local_path = self.local_dir / data_type / f"{identifier}.csv"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(file_path, str(local_path))
        logger.info(f"Saved CSV locally: {local_path}")

        return str(local_path)


class LeaguesCollector:
    """Collects league and tournament metadata."""

    def __init__(self, storage: BronzeStorage):
        self.storage = storage

    async def collect(self) -> dict[str, Any]:
        """Collect all leagues and their tournaments."""
        results = {"leagues": [], "tournaments": [], "errors": []}

        async with LoLEsportsClient() as client:
            try:
                leagues = await client.get_leagues()
                logger.info(f"Fetched {len(leagues)} leagues")

                for league in leagues:
                    league_dict = league.model_dump(mode="json")
                    results["leagues"].append(league_dict)

                    self.storage.save_json(
                        data=league_dict,
                        data_type="leagues",
                        identifier=league.slug,
                    )

                    try:
                        tournaments = await client.get_tournaments(league.id)
                        logger.info(f"Fetched {len(tournaments)} tournaments for {league.name}")

                        for tournament in tournaments:
                            tournament_dict = tournament.model_dump(mode="json")
                            results["tournaments"].append(tournament_dict)

                            self.storage.save_json(
                                data=tournament_dict,
                                data_type="tournaments",
                                identifier=f"{league.slug}_{tournament.slug}",
                            )

                    except Exception as e:
                        error_msg = f"Error fetching tournaments for {league.name}: {e}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)

            except Exception as e:
                error_msg = f"Error fetching leagues: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        return results


class MatchesCollector:
    """Collects match data for specific leagues."""

    def __init__(self, storage: BronzeStorage):
        self.storage = storage

    async def collect(
        self,
        league_ids: list[str],
        include_game_details: bool = False,
    ) -> dict[str, Any]:
        """
        Collect completed matches for specified leagues.

        Args:
            league_ids: List of league IDs to collect
            include_game_details: Whether to fetch detailed game data
        """
        results = {"matches": [], "games": [], "errors": []}

        async with LoLEsportsClient() as api_client:
            for league_id in league_ids:
                try:
                    matches = await api_client.get_completed_matches_raw(league_id)
                    logger.info(f"Fetched {len(matches)} matches for league {league_id}")

                    for match in matches:
                        results["matches"].append(match)

                        start_time = None
                        if match.get("start_time"):
                            try:
                                start_time = datetime.fromisoformat(
                                    match["start_time"].replace("Z", "+00:00")
                                )
                            except (ValueError, TypeError):
                                pass

                        self.storage.save_json(
                            data=match,
                            data_type="matches",
                            identifier=match["match_id"],
                            timestamp=start_time,
                        )

                except Exception as e:
                    error_msg = f"Error fetching matches for league {league_id}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)

        return results


class LiveStatsCollector:
    """Collects detailed frame-by-frame game stats."""

    def __init__(self, storage: BronzeStorage):
        self.storage = storage

    async def collect(self, game_ids: list[str]) -> dict[str, Any]:
        """
        Collect live stats for specified games.

        Args:
            game_ids: List of game IDs to fetch detailed stats for
        """
        results = {"games": [], "errors": []}

        async with LoLEsportsFeedClient() as feed_client:
            for game_id in game_ids:
                try:
                    game_data = await feed_client.get_full_game_data(game_id)

                    if game_data.get("frames"):
                        results["games"].append(game_data)

                        self.storage.save_json(
                            data=game_data,
                            data_type="live_stats",
                            identifier=game_id,
                        )

                        logger.info(
                            f"Collected {len(game_data.get('frames', []))} frames "
                            f"and {len(game_data.get('events', []))} events for game {game_id}"
                        )

                except Exception as e:
                    error_msg = f"Error fetching live stats for game {game_id}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)

        return results


class OracleElixirCollector:
    """Collects historical data from Oracle's Elixir."""

    def __init__(self, storage: BronzeStorage):
        self.storage = storage
        self.downloader = OracleElixirDownloader()

    async def collect(
        self,
        years: list[str] | None = None,
        leagues: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Download and store Oracle's Elixir data.

        Args:
            years: List of years to download (defaults to recent years)
            leagues: Optional list of leagues to filter
        """
        if years is None:
            years = ["2024", "2023", "2022"]

        results = await self.downloader.download_all(years=years, filter_leagues=leagues)

        summary = {"downloaded": [], "errors": []}

        for result in results:
            if result.success and result.file_path:
                s3_path = self.storage.save_csv(
                    file_path=result.file_path,
                    data_type="oracle_elixir",
                    identifier=f"oracle_elixir_{result.year}",
                )
                summary["downloaded"].append(
                    {"year": result.year, "rows": result.rows, "path": s3_path}
                )
            elif result.error:
                summary["errors"].append({"year": result.year, "error": result.error})

        return summary


async def run_full_ingestion(
    config: CollectorConfig | None = None,
    leagues_filter: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run full data ingestion pipeline.

    Args:
        config: Collector configuration
        leagues_filter: Optional list of league slugs to focus on
    """
    config = config or CollectorConfig()
    storage = BronzeStorage(config)

    results = {
        "leagues": None,
        "matches": None,
        "oracle_elixir": None,
        "started_at": datetime.utcnow().isoformat(),
        "finished_at": None,
    }

    logger.info("Starting leagues collection...")
    leagues_collector = LeaguesCollector(storage)
    results["leagues"] = await leagues_collector.collect()

    if leagues_filter:
        league_ids = [
            league["id"]
            for league in results["leagues"]["leagues"]
            if league["slug"].upper() in [l.upper() for l in leagues_filter]
        ]
    else:
        league_ids = [league["id"] for league in results["leagues"]["leagues"]]

    logger.info(f"Starting matches collection for {len(league_ids)} leagues...")
    matches_collector = MatchesCollector(storage)
    results["matches"] = await matches_collector.collect(league_ids[:5])

    logger.info("Starting Oracle's Elixir collection...")
    oracle_collector = OracleElixirCollector(storage)
    results["oracle_elixir"] = await oracle_collector.collect(
        years=["2024"], leagues=leagues_filter
    )

    results["finished_at"] = datetime.utcnow().isoformat()

    logger.info("Ingestion completed!")
    return results


async def main():
    """Example usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = CollectorConfig(
        local_backup_dir="data/bronze"
    )

    results = await run_full_ingestion(
        config=config,
        leagues_filter=["CBLOL", "LCK", "LEC"],
    )

    print("\n=== Ingestion Results ===")
    print(f"Leagues collected: {len(results['leagues']['leagues'])}")
    print(f"Tournaments collected: {len(results['leagues']['tournaments'])}")
    print(f"Matches collected: {len(results['matches']['matches'])}")
    print(f"Oracle's Elixir: {results['oracle_elixir']}")


if __name__ == "__main__":
    asyncio.run(main())
