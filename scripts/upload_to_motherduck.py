"""Upload Silver Parquet tables to MotherDuck."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

from src.upcoming_matches import load_upcoming_matches

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def upload_to_motherduck(token: str, db_name: str = "lolesports"):
    """
    Connect to MotherDuck and upload local Parquet tables.
    """
    logger.info(f"Connecting to MotherDuck database: {db_name}...")
    
    # Connect using the token to the default database first
    con_str = f"md:?motherduck_token={token}"
    
    try:
        con = duckdb.connect(con_str)
        logger.info("Connected to MotherDuck successfully!")
        
        # Create database if it doesn't exist and use it
        logger.info(f"Creating database '{db_name}' if it doesn't exist...")
        con.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        con.execute(f"USE {db_name}")
        
        # Define tables to upload based on local Bronze/Silver directories
        silver_dir = Path("data/silver")
        
        if not silver_dir.exists():
            logger.error(f"Silver directory not found at {silver_dir}. Run processing first.")
            return False
            
        # 1. Upload Players Table
        players_dir = silver_dir / "players"
        if players_dir.exists() and any(players_dir.iterdir()):
            players_path = players_dir / "**" / "*.parquet"
            logger.info("Creating 'players' table in MotherDuck...")
            con.execute(f"CREATE OR REPLACE TABLE players AS SELECT * FROM '{players_path}'")
        else:
            logger.info("Skipping 'players' table upload (no local data).")
        
        # 2. Upload Games Table
        games_dir = silver_dir / "games"
        if games_dir.exists() and any(games_dir.iterdir()):
            games_path = games_dir / "**" / "*.parquet"
            logger.info("Creating 'games' table in MotherDuck...")
            con.execute(f"CREATE OR REPLACE TABLE games AS SELECT * FROM '{games_path}'")
        else:
            logger.info("Skipping 'games' table upload (no local data).")
        
        # 3. Upload Series Table
        series_path = silver_dir / "series.parquet"
        if series_path.exists():
            logger.info("Creating 'series' table in MotherDuck...")
            con.execute(f"CREATE OR REPLACE TABLE series AS SELECT * FROM '{series_path}'")
            
        # 4. Upload Champions Table
        champs_path = silver_dir / "champions.parquet"
        if champs_path.exists():
            logger.info("Creating 'champions' table in MotherDuck...")
            con.execute(f"CREATE OR REPLACE TABLE champions AS SELECT * FROM '{champs_path}'")

        # 5. Upload latest Gold snapshot tables when available
        gold_pointer = Path("data/gold/latest_snapshot.json")
        if gold_pointer.exists():
            pointer = json.loads(gold_pointer.read_text(encoding="utf-8"))
            snapshot_dir = Path(pointer["snapshot_dir"])
            logger.info(f"Uploading Gold snapshot from {snapshot_dir}...")

            gold_tables = {
                "gold_dim_league": snapshot_dir / "dim_league.parquet",
                "gold_dim_team": snapshot_dir / "dim_team.parquet",
                "gold_dim_player": snapshot_dir / "dim_player.parquet",
                "gold_external_reconciliation": snapshot_dir / "external_reconciliation.parquet",
                "gold_fact_game_team": snapshot_dir / "fact_game_team.parquet",
                "gold_fact_game_player": snapshot_dir / "fact_game_player.parquet",
                "gold_fact_draft": snapshot_dir / "fact_draft.parquet",
                "gold_fact_series": snapshot_dir / "fact_series.parquet",
                "gold_match_features_prematch": snapshot_dir / "match_features_prematch.parquet",
                "gold_model_core_series": snapshot_dir / "model_core_series.parquet",
                "gold_quality_issues": snapshot_dir / "quality_issues.parquet",
                "gold_source_coverage": snapshot_dir / "source_coverage.parquet",
                "gold_validation_summary": snapshot_dir / "validation_summary.parquet",
                "gold_dataset_manifest": snapshot_dir / "dataset_manifest.parquet",
            }

            for table_name, table_path in gold_tables.items():
                if table_path.exists():
                    logger.info(f"Creating '{table_name}' table in MotherDuck...")
                    con.execute(
                        f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM '{table_path}'"
                    )
                else:
                    logger.info(f"Skipping '{table_name}' upload (missing file: {table_path.name}).")
        else:
            logger.info("Skipping Gold upload (data/gold/latest_snapshot.json not found).")

        # 6. Upload normalized upcoming match schedule for the web app
        upcoming_df, upcoming_meta = load_upcoming_matches()
        logger.info(
            "Creating 'web_upcoming_matches' table in MotherDuck with %s upcoming rows...",
            len(upcoming_df),
        )
        con.register("web_upcoming_matches_df", upcoming_df)
        con.execute("CREATE OR REPLACE TABLE web_upcoming_matches AS SELECT * FROM web_upcoming_matches_df")

        logger.info("Creating 'web_upcoming_matches_meta' table in MotherDuck...")
        con.register(
            "web_upcoming_matches_meta_df",
            pd.DataFrame(
                [
                    {
                        "fetched_at": upcoming_meta.get("fetched_at"),
                        "row_count": int(upcoming_meta.get("row_count", 0)),
                        "source_path": str(upcoming_meta.get("path", "")),
                    }
                ]
            ),
        )
        con.execute(
            "CREATE OR REPLACE TABLE web_upcoming_matches_meta AS SELECT * FROM web_upcoming_matches_meta_df"
        )

        # Verify
        logger.info("Verifying tables...")
        tables = con.execute("SHOW TABLES").fetchall()
        logger.info(f"Tables in MotherDuck: {[t[0] for t in tables]}")
        
        logger.info("Upload complete!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to upload to MotherDuck: {e}")
        return False
    finally:
        if 'con' in locals():
            con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Silver data to MotherDuck")
    parser.add_argument("--token", type=str, help="MotherDuck API Token (or use MOTHERDUCK_TOKEN env var)")
    parser.add_argument("--db", type=str, default="lolesports", help="MotherDuck Database Name")
    
    args = parser.parse_args()
    
    token = args.token or os.environ.get("MOTHERDUCK_TOKEN")
    
    if not token:
        logger.error("MotherDuck token not provided. Use --token or MOTHERDUCK_TOKEN env var.")
        sys.exit(1)
        
    success = upload_to_motherduck(token=token, db_name=args.db)
    sys.exit(0 if success else 1)
