"""Run Silver layer transformation."""

import sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding='utf-8')

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.transform.silver_matches import SilverMatchesTransformer


def main():
    print("=" * 70)
    print("SILVER LAYER TRANSFORMATION")
    print("=" * 70)
    
    transformer = SilverMatchesTransformer()
    files = transformer.run()
    
    print("\n" + "=" * 70)
    print("ARQUIVOS GERADOS:")
    print("=" * 70)
    for name, path in files.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {name}: {path} ({size_mb:.1f} MB)")
    
    # Quick validation
    print("\n" + "=" * 70)
    print("VALIDACAO - CAMPEOES DE WORLDS")
    print("=" * 70)
    
    import duckdb
    con = duckdb.connect(":memory:")
    
    query = """
    SELECT year, champion, runner_up, final_score
    FROM read_parquet('data/silver/champions.parquet')
    WHERE league = 'WLDs'
    ORDER BY year
    """
    result = con.execute(query).fetchdf()
    print(result.to_string())
    
    print("\n" + "=" * 70)
    print("VALIDACAO - CAMPEOES DE MSI")
    print("=" * 70)
    
    query = """
    SELECT year, champion, runner_up, final_score
    FROM read_parquet('data/silver/champions.parquet')
    WHERE league = 'MSI'
    ORDER BY year
    """
    result = con.execute(query).fetchdf()
    print(result.to_string())


if __name__ == "__main__":
    main()
