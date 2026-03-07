"""Upload Silver Parquet tables to MotherDuck."""

import argparse
import logging
import os
import sys
from pathlib import Path

import duckdb

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
