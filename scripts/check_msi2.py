"""Check MSI winners - all games."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")
csv = "data/bronze/oracle_elixir/*_LoL_esports_match_data_from_OraclesElixir.csv"

print("=" * 70)
print("MSI - TODOS OS RESULTADOS")
print("=" * 70)

# Buscar todos os times do MSI por ano
query = f"""
SELECT 
    year,
    teamname,
    COUNT(*) as games,
    SUM(result) as wins,
    ROUND(100.0 * SUM(result) / COUNT(*), 1) as winrate
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE LOWER(league) LIKE '%msi%'
GROUP BY year, teamname
ORDER BY year, wins DESC
"""

result = con.execute(query).fetchdf()

print("\nMSI por ano (ordenado por vitórias):")
print("-" * 70)

current_year = None
for _, row in result.iterrows():
    year = int(row['year'])
    if year != current_year:
        current_year = year
        print(f"\n{year} MSI:")
    wins = int(row['wins'])
    games = int(row['games'])
    wr = row['winrate']
    # O time com mais vitórias provavelmente é o campeão
    marker = " <-- CAMPEÃO" if wins == result[result['year'] == year]['wins'].max() else ""
    print(f"  {row['teamname']}: {wins}W-{games-wins}L ({wr}%){marker}")
