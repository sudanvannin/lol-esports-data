"""Precompute upcoming match probabilities and upload them to MotherDuck."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime

import duckdb
import pandas as pd

from web import db, prob_win

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _upload_dataframe(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    df: pd.DataFrame,
) -> None:
    view_name = f"{table_name}_df"
    con.register(view_name, df)
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {view_name}")
    con.unregister(view_name)


def build_prob_win_tables(
    limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    upcoming_df = db.get_upcoming_matches(limit=limit)
    rows = upcoming_df.to_dict("records")

    match_records: list[dict] = []
    totals_records: list[dict] = []
    failures = 0

    for row in rows:
        try:
            detail = prob_win.build_prob_win_detail(row)
            match_record, market_rows = prob_win.flatten_prob_win_detail(detail)
        except Exception as exc:
            failures += 1
            logger.warning(
                "Failed to score %s vs %s (%s): %s",
                row.get("team1"),
                row.get("team2"),
                row.get("match_id"),
                exc,
            )
            context = prob_win.build_match_context(row)
            match_record = {
                "match_id": row.get("match_id"),
                "match_time": row.get("match_time"),
                "match_date": row.get("match_date"),
                "league": row.get("league"),
                "league_label": row.get("league_label"),
                "event_name": row.get("event_name"),
                "phase_label": row.get("phase_label"),
                "team1": row.get("team1"),
                "team2": row.get("team2"),
                "best_of": context.get("best_of"),
                "patch": context.get("patch_version"),
                "league_code": context.get("league_code"),
                "split_name": context.get("split_name"),
                "playoffs": context.get("playoffs"),
                "winner_available": False,
                "winner_error": str(exc),
                "team1_win_prob": None,
                "team2_win_prob": None,
                "team1_win_pct": None,
                "team2_win_pct": None,
                "team1_fair_odds": None,
                "team2_fair_odds": None,
                "favorite_name": None,
                "favorite_prob_pct": None,
                "totals_available": False,
                "totals_error": str(exc),
                "warnings_json": "[]",
            }
            market_rows = []

        match_records.append(match_record)
        totals_records.extend(market_rows)

    generated_at = datetime.now(UTC).isoformat()
    matches_df = pd.DataFrame(match_records)
    totals_df = pd.DataFrame(totals_records)
    meta_df = pd.DataFrame(
        [
            {
                "generated_at": generated_at,
                "match_count": int(len(matches_df)),
                "totals_row_count": int(len(totals_df)),
                "failed_match_count": int(failures),
                "source": "local_model_scoring",
            }
        ]
    )
    return matches_df, totals_df, meta_df


def upload_prob_win_tables(token: str, db_name: str, limit: int) -> int:
    matches_df, totals_df, meta_df = build_prob_win_tables(limit=limit)
    logger.info(
        "Uploading %s match quotes and %s totals rows to %s.",
        len(matches_df),
        len(totals_df),
        db_name,
    )

    con = duckdb.connect(f"md:?motherduck_token={token}")
    try:
        con.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        con.execute(f"USE {db_name}")
        _upload_dataframe(con, "web_prob_win_matches", matches_df)
        _upload_dataframe(con, "web_prob_win_totals", totals_df)
        _upload_dataframe(con, "web_prob_win_meta", meta_df)
    finally:
        con.close()

    logger.info("Prob-win tables uploaded successfully.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precompute upcoming match probabilities and upload them to MotherDuck."
    )
    parser.add_argument("--token", type=str, help="MotherDuck token")
    parser.add_argument("--db", type=str, default="lolesports")
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    token = args.token or os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        logger.error("MotherDuck token not provided.")
        return 1

    return upload_prob_win_tables(token=token, db_name=args.db, limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
