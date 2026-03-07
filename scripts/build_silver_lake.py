"""Build complete Silver lake with all tables."""

import sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding='utf-8')

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from src.transform.silver_matches import SilverMatchesTransformer
from src.transform.silver_games import SilverGamesTransformer


def main():
    print("=" * 70)
    print("BUILDING SILVER LAKEHOUSE")
    print("=" * 70)
    
    # 1. Games and Players tables
    print("\n[1/2] Creating games and players tables...")
    games_transformer = SilverGamesTransformer()
    games_transformer.run()
    
    # 2. Series and Champions tables (already exists, refresh)
    print("\n[2/2] Creating series and champions tables...")
    matches_transformer = SilverMatchesTransformer()
    matches_transformer.run()
    
    # Summary
    print("\n" + "=" * 70)
    print("SILVER LAKE COMPLETE")
    print("=" * 70)
    
    silver_path = Path("data/silver")
    
    if not silver_path.exists() or not any(silver_path.iterdir()):
        print("\nNo tables were created (Silver directory empty or missing). Skipping summary and tests.")
        return
        
    print("\nTabelas criadas:")
    for table_dir in silver_path.iterdir():
        if table_dir.is_dir():
            # Count partitions
            partitions = list(table_dir.glob("**/*.parquet"))
            if not partitions:
                continue
            total_size = sum(p.stat().st_size for p in partitions) / (1024 * 1024)
            print(f"  - {table_dir.name}: {len(partitions)} partitions, {total_size:.1f} MB")
        elif table_dir.suffix == '.parquet':
            size = table_dir.stat().st_size / (1024 * 1024)
            print(f"  - {table_dir.stem}: {size:.1f} MB")
    
    # Test query
    if list(silver_path.glob("players/**/*.parquet")):
        print("\n" + "=" * 70)
        print("TESTE: Chovy no MSI 2025")
        print("=" * 70)
        
        import duckdb
        con = duckdb.connect(":memory:")
        
        query = """
        SELECT 
            game_date,
            game,
            champion,
            kills,
            deaths,
            assists,
            result
        FROM read_parquet('data/silver/players/**/*.parquet')
        WHERE LOWER(playername) = 'chovy'
            AND league = 'MSI'
            AND year = 2025
        ORDER BY game_date DESC, game
        LIMIT 10
        """
        result = con.execute(query).fetchdf()
        print(result.to_string(index=False))
    else:
        print("\nSkipping test query (no players parquet data found).")


if __name__ == "__main__":
    main()
