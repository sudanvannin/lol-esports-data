"""DAG for ingesting LoL Esports data from APIs."""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)

default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

LEAGUES_TO_COLLECT = ["cblol-brazil", "lck", "lec", "lcs", "lpl", "worlds", "msi"]


def collect_leagues(**context):
    """Collect all leagues and tournaments."""
    from ingestion.collectors import BronzeStorage, CollectorConfig, LeaguesCollector

    config = CollectorConfig()
    storage = BronzeStorage(config)
    collector = LeaguesCollector(storage)

    results = asyncio.run(collector.collect())

    logger.info(f"Collected {len(results['leagues'])} leagues")
    logger.info(f"Collected {len(results['tournaments'])} tournaments")

    if results["errors"]:
        logger.warning(f"Errors: {results['errors']}")

    context["ti"].xcom_push(key="leagues", value=results["leagues"])

    return results


def collect_matches(**context):
    """Collect matches for configured leagues."""
    from ingestion.collectors import (
        BronzeStorage,
        CollectorConfig,
        MatchesCollector,
    )

    leagues = context["ti"].xcom_pull(key="leagues", task_ids="ingest_leagues")

    if not leagues:
        logger.warning("No leagues found in XCom, fetching fresh...")
        from ingestion.collectors import LeaguesCollector

        config = CollectorConfig()
        storage = BronzeStorage(config)
        leagues_collector = LeaguesCollector(storage)
        leagues_result = asyncio.run(leagues_collector.collect())
        leagues = leagues_result["leagues"]

    league_ids = [
        league["id"]
        for league in leagues
        if league["slug"].upper() in [l.upper() for l in LEAGUES_TO_COLLECT]
    ]

    logger.info(f"Collecting matches for {len(league_ids)} leagues")

    config = CollectorConfig()
    storage = BronzeStorage(config)
    collector = MatchesCollector(storage)

    results = asyncio.run(collector.collect(league_ids, include_game_details=False))

    logger.info(f"Collected {len(results['matches'])} matches")

    game_ids = []
    for match in results["matches"]:
        for game in match.get("games", []):
            if game.get("id"):
                game_ids.append(game["id"])

    context["ti"].xcom_push(key="game_ids", value=game_ids[:100])

    return results


def collect_game_details(**context):
    """Collect detailed stats for games."""
    from ingestion.collectors import BronzeStorage, CollectorConfig, LiveStatsCollector

    game_ids = context["ti"].xcom_pull(key="game_ids", task_ids="ingest_matches")

    if not game_ids:
        logger.info("No game IDs to collect details for")
        return {"games": [], "errors": []}

    logger.info(f"Collecting details for {len(game_ids)} games")

    config = CollectorConfig()
    storage = BronzeStorage(config)
    collector = LiveStatsCollector(storage)

    results = asyncio.run(collector.collect(game_ids))

    logger.info(f"Collected details for {len(results['games'])} games")

    return results


with DAG(
    dag_id="lol_esports_ingestion",
    default_args=default_args,
    description="Ingest LoL Esports data from APIs to Bronze layer",
    schedule_interval="0 */6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["lol", "ingestion", "bronze"],
) as dag:
    start = EmptyOperator(task_id="start")

    ingest_leagues = PythonOperator(
        task_id="ingest_leagues",
        python_callable=collect_leagues,
    )

    ingest_matches = PythonOperator(
        task_id="ingest_matches",
        python_callable=collect_matches,
    )

    ingest_game_details = PythonOperator(
        task_id="ingest_game_details",
        python_callable=collect_game_details,
    )

    end = EmptyOperator(task_id="end")

    start >> ingest_leagues >> ingest_matches >> ingest_game_details >> end
