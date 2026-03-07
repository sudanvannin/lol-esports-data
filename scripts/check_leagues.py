"""Check available leagues in data."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")
csv = "data/bronze/oracle_elixir/*_LoL_esports_match_data_from_OraclesElixir.csv"

query = f"""
SELECT DISTINCT
    league,
    COUNT(*) as games
FROM read_csv_auto('{csv}', ignore_errors=true)
GROUP BY league
ORDER BY games DESC
"""

result = con.execute(query).fetchdf()

print("Ligas disponíveis nos dados:")
print("-" * 50)
for _, row in result.iterrows():
    print(f"  {row['league']}: {row['games']} jogos")
