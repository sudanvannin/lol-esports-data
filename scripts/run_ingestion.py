"""
Local ingestion script for Bronze layer.
Runs data collection without Docker/MinIO dependencies.

Usage:
    python scripts/run_ingestion.py [--leagues-only] [--oracle-only] [--full]
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.collectors import (
    BronzeStorage,
    CollectorConfig,
    LeaguesCollector,
    MatchesCollector,
    OracleElixirCollector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingestion")


LEAGUES_TO_COLLECT = [
    "cblol-brazil",
    "lck",
    "lec",
    "lcs",
    "lpl",
    "worlds",
    "msi",
]


async def collect_leagues(storage: BronzeStorage) -> dict:
    """Collect all leagues and tournaments."""
    logger.info("=" * 60)
    logger.info("COLLECTING LEAGUES AND TOURNAMENTS")
    logger.info("=" * 60)

    collector = LeaguesCollector(storage)
    results = await collector.collect()

    logger.info(f"Leagues collected: {len(results['leagues'])}")
    logger.info(f"Tournaments collected: {len(results['tournaments'])}")

    if results["errors"]:
        logger.warning(f"Errors: {len(results['errors'])}")
        for error in results["errors"]:
            logger.warning(f"  - {error}")

    return results


async def collect_matches(
    storage: BronzeStorage,
    leagues_data: list[dict],
    target_leagues: list[str],
) -> dict:
    """Collect matches for specified leagues."""
    logger.info("=" * 60)
    logger.info("COLLECTING MATCHES")
    logger.info("=" * 60)

    league_ids = []
    for league in leagues_data:
        slug = league.get("slug", "").lower()
        if slug in [l.lower() for l in target_leagues]:
            league_ids.append(league["id"])
            logger.info(f"  Including: {league['name']} ({slug})")

    if not league_ids:
        logger.warning("No matching leagues found!")
        return {"matches": [], "games": [], "errors": []}

    collector = MatchesCollector(storage)
    results = await collector.collect(
        league_ids=league_ids,
        include_game_details=False,
    )

    logger.info(f"Matches collected: {len(results['matches'])}")
    logger.info(f"Games collected: {len(results['games'])}")

    if results["errors"]:
        logger.warning(f"Errors: {len(results['errors'])}")

    return results


async def collect_oracle_elixir(storage: BronzeStorage, years: list[str]) -> dict:
    """Download Oracle's Elixir historical data."""
    logger.info("=" * 60)
    logger.info("COLLECTING ORACLE'S ELIXIR DATA")
    logger.info("=" * 60)

    collector = OracleElixirCollector(storage)
    results = await collector.collect(years=years, leagues=None)

    logger.info(f"Files downloaded: {len(results.get('downloaded', []))}")

    if results.get("errors"):
        logger.warning(f"Errors: {len(results['errors'])}")

    return results


async def run_full_ingestion(args: argparse.Namespace) -> dict:
    """Run the full ingestion pipeline."""
    start_time = datetime.now()

    logger.info("=" * 60)
    logger.info("LOL ESPORTS DATA INGESTION")
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    config = CollectorConfig(
        local_backup_dir="data/bronze",
    )
    storage = BronzeStorage(config)

    results = {
        "started_at": start_time.isoformat(),
        "leagues": None,
        "matches": None,
        "oracle_elixir": None,
        "finished_at": None,
        "duration_seconds": None,
    }

    if args.leagues_only or args.full:
        results["leagues"] = await collect_leagues(storage)

    if args.matches and results.get("leagues"):
        results["matches"] = await collect_matches(
            storage,
            results["leagues"]["leagues"],
            LEAGUES_TO_COLLECT,
        )

    if args.oracle_only or args.full:
        results["oracle_elixir"] = await collect_oracle_elixir(
            storage, years=["2024", "2023"]
        )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    results["finished_at"] = end_time.isoformat()
    results["duration_seconds"] = duration

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info("=" * 60)

    return results


def print_summary(results: dict):
    """Print a summary of collected data."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results.get("leagues"):
        print(f"  Leagues: {len(results['leagues']['leagues'])}")
        print(f"  Tournaments: {len(results['leagues']['tournaments'])}")

    if results.get("matches"):
        print(f"  Matches: {len(results['matches']['matches'])}")
        print(f"  Games: {len(results['matches']['games'])}")

    if results.get("oracle_elixir"):
        downloaded = results["oracle_elixir"].get("downloaded", [])
        print(f"  Oracle's Elixir files: {len(downloaded)}")

    print(f"\n  Duration: {results.get('duration_seconds', 0):.1f}s")
    print(f"  Output: data/bronze/")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="LoL Esports Data Ingestion to Bronze Layer"
    )
    parser.add_argument(
        "--leagues-only",
        action="store_true",
        help="Only collect leagues and tournaments",
    )
    parser.add_argument(
        "--matches",
        action="store_true",
        help="Also collect matches (requires leagues)",
    )
    parser.add_argument(
        "--oracle-only",
        action="store_true",
        help="Only download Oracle's Elixir data",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full ingestion (leagues + oracle)",
    )

    args = parser.parse_args()

    if not any([args.leagues_only, args.oracle_only, args.full]):
        args.leagues_only = True

    try:
        results = asyncio.run(run_full_ingestion(args))
        print_summary(results)
    except KeyboardInterrupt:
        logger.info("\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
