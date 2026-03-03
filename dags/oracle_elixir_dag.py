"""DAG for ingesting historical data from Oracle's Elixir."""

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
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

YEARS_TO_COLLECT = ["2024", "2023", "2022"]
LEAGUES_FILTER = ["CBLOL", "LCK", "LEC", "LCS", "LPL"]  # Oracle uses display names, not slugs


def download_oracle_elixir(**context):
    """Download Oracle's Elixir CSV files."""
    from ingestion.collectors import (
        BronzeStorage,
        CollectorConfig,
        OracleElixirCollector,
    )

    config = CollectorConfig()
    storage = BronzeStorage(config)
    collector = OracleElixirCollector(storage)

    results = asyncio.run(
        collector.collect(years=YEARS_TO_COLLECT, leagues=LEAGUES_FILTER)
    )

    logger.info(f"Downloaded: {results['downloaded']}")

    if results["errors"]:
        logger.warning(f"Errors: {results['errors']}")

    return results


with DAG(
    dag_id="oracle_elixir_ingestion",
    default_args=default_args,
    description="Download historical pro play data from Oracle's Elixir",
    schedule_interval="0 0 * * 0",  # Weekly on Sunday
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["lol", "ingestion", "bronze", "historical"],
) as dag:
    start = EmptyOperator(task_id="start")

    download_data = PythonOperator(
        task_id="download_oracle_elixir",
        python_callable=download_oracle_elixir,
    )

    end = EmptyOperator(task_id="end")

    start >> download_data >> end
