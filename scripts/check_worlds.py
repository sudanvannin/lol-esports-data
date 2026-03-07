"""Check Worlds winners."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")
csv = "data/bronze/oracle_elixir/*_LoL_esports_match_data_from_OraclesElixir.csv"

print("=" * 70)
print("WORLDS - VERIFICANDO CAMPEÕES")
print("=" * 70)

query = f"""
SELECT 
    year,
    teamname,
    COUNT(*) as games,
    SUM(result) as wins,
    ROUND(100.0 * SUM(result) / COUNT(*), 1) as winrate
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE LOWER(league) LIKE '%worlds%'
GROUP BY year, teamname
ORDER BY year, wins DESC
"""

result = con.execute(query).fetchdf()

print("\nWorlds por ano (ordenado por vitórias):")
print("-" * 70)

current_year = None
count = 0
for _, row in result.iterrows():
    year = int(row['year'])
    if year != current_year:
        current_year = year
        count = 0
        print(f"\n{year} Worlds:")
    
    count += 1
    if count <= 5:  # Top 5 times
        wins = int(row['wins'])
        games = int(row['games'])
        wr = row['winrate']
        print(f"  {row['teamname']}: {wins}W-{games-wins}L ({wr}%)")
