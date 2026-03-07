"""Quick query for series data."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")

# Get args
league = sys.argv[1] if len(sys.argv) > 1 else "WLDs"
year = int(sys.argv[2]) if len(sys.argv) > 2 else 2016
phase = sys.argv[3] if len(sys.argv) > 3 else "Semifinal"

query = f"""
SELECT 
    match_date,
    team1,
    team2,
    score,
    series_winner,
    series_format
FROM read_parquet('data/silver/series.parquet')
WHERE league = '{league}' AND year = {year} AND tournament_phase = '{phase}'
ORDER BY match_date
"""

result = con.execute(query).fetchdf()
print(f"\n{league} {year} - {phase}:")
print("-" * 70)
print(result.to_string(index=False))
