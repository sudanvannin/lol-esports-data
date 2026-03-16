"""Refresh fast web tables from the official Riot schedule and upload them."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import duckdb
import pandas as pd

from src.official_schedule import (
    DEFAULT_WEB_SCHEDULE_PATH,
    collect_official_schedule_snapshot,
    save_official_schedule_snapshot,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
SUPPORTED_MOTHERDUCK_DUCKDB_VERSION = "1.4.4"


def _upload_dataframe(
    con: duckdb.DuckDBPyConnection, table_name: str, df: pd.DataFrame
):
    view_name = f"{table_name}_df"
    con.register(view_name, df)
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {view_name}")
    con.unregister(view_name)


def upload_fast_web_tables(
    *,
    token: str,
    db_name: str,
    recent_df: pd.DataFrame,
    upcoming_df: pd.DataFrame,
    metadata: dict,
) -> None:
    """Upload fast-lane web tables to MotherDuck."""
    installed_duckdb_version = getattr(duckdb, "__version__", "unknown")
    if installed_duckdb_version != SUPPORTED_MOTHERDUCK_DUCKDB_VERSION:
        raise RuntimeError(
            "MotherDuck fast-lane refresh requires duckdb=="
            f"{SUPPORTED_MOTHERDUCK_DUCKDB_VERSION}, got {installed_duckdb_version}."
        )

    con = duckdb.connect(f"md:?motherduck_token={token}")
    try:
        con.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        con.execute(f"USE {db_name}")

        _upload_dataframe(con, "web_recent_matches_live", recent_df)
        _upload_dataframe(con, "web_upcoming_matches_live", upcoming_df)

        recent_meta_df = pd.DataFrame(
            [
                {
                    "fetched_at": metadata.get("fetched_at"),
                    "row_count": int(metadata.get("recent_row_count", len(recent_df))),
                    "event_count": int(metadata.get("event_count", 0)),
                    "pages_fetched": int(metadata.get("pages_fetched", 0)),
                    "lookback_days": int(metadata.get("recent_lookback_days", 0)),
                    "source_path": str(metadata.get("path", DEFAULT_WEB_SCHEDULE_PATH)),
                    "source": metadata.get("source", "riot_official_schedule"),
                }
            ]
        )
        _upload_dataframe(con, "web_recent_matches_live_meta", recent_meta_df)

        upcoming_meta_df = pd.DataFrame(
            [
                {
                    "fetched_at": metadata.get("fetched_at"),
                    "row_count": int(
                        metadata.get("upcoming_row_count", len(upcoming_df))
                    ),
                    "event_count": int(metadata.get("event_count", 0)),
                    "pages_fetched": int(metadata.get("pages_fetched", 0)),
                    "lookback_days": int(metadata.get("recent_lookback_days", 0)),
                    "source_path": str(metadata.get("path", DEFAULT_WEB_SCHEDULE_PATH)),
                    "source": metadata.get("source", "riot_official_schedule"),
                }
            ]
        )
        _upload_dataframe(con, "web_upcoming_matches_live_meta", upcoming_meta_df)
    finally:
        con.close()


async def _run(args: argparse.Namespace) -> int:
    recent_df, upcoming_df, metadata = await collect_official_schedule_snapshot(
        max_pages=args.max_pages,
        recent_limit=args.recent_limit,
        upcoming_limit=args.upcoming_limit,
        recent_lookback_days=args.lookback_days,
    )
    save_official_schedule_snapshot(
        recent_df,
        upcoming_df,
        metadata,
        path=DEFAULT_WEB_SCHEDULE_PATH,
    )

    logger.info(
        "Built official schedule snapshot with %s recent matches and %s upcoming matches.",
        len(recent_df),
        len(upcoming_df),
    )

    token = args.token or os.environ.get("MOTHERDUCK_TOKEN")
    if token:
        upload_fast_web_tables(
            token=token,
            db_name=args.db,
            recent_df=recent_df,
            upcoming_df=upcoming_df,
            metadata=metadata,
        )
        logger.info("Uploaded fast web tables to MotherDuck database %s.", args.db)
    else:
        logger.warning(
            "MOTHERDUCK_TOKEN not provided. Saved the local snapshot only at %s.",
            DEFAULT_WEB_SCHEDULE_PATH,
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh fast web tables from the official Riot schedule."
    )
    parser.add_argument("--token", type=str, help="MotherDuck token")
    parser.add_argument("--db", type=str, default="lolesports")
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--recent-limit", type=int, default=80)
    parser.add_argument("--upcoming-limit", type=int, default=80)
    parser.add_argument("--lookback-days", type=int, default=7)
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
