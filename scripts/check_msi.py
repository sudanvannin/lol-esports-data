"""Check MSI winners."""

import duckdb

con = duckdb.connect(":memory:")
csv = "data/bronze/oracle_elixir/*_LoL_esports_match_data_from_OraclesElixir.csv"

print("=" * 70)
print("MSI - VERIFICANDO CAMPEÕES")
print("=" * 70)

# Buscar times com mais vitórias em MSI por ano
query = f"""
SELECT 
    year,
    teamname,
    COUNT(*) as games,
    SUM(result) as wins,
    ROUND(100.0 * SUM(result) / COUNT(*), 1) as winrate
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE LOWER(league) LIKE '%msi%'
  AND playoffs = 1
GROUP BY year, teamname
HAVING SUM(result) > 0
ORDER BY year, wins DESC
"""

result = con.execute(query).fetchdf()

print("\nMSI Playoffs por ano (times com vitórias):")
print("-" * 70)

current_year = None
for _, row in result.iterrows():
    year = int(row['year'])
    if year != current_year:
        current_year = year
        print(f"\n{year} MSI:")
    print(f"  {row['teamname']}: {int(row['wins'])} vitórias em {int(row['games'])} jogos ({row['winrate']}%)")

# Verificar finais específicas
print("\n" + "=" * 70)
print("FINAIS MSI (últimos jogos do torneio)")
print("=" * 70)

for year in [2022, 2023, 2024, 2025]:
    query = f"""
    SELECT 
        teamname,
        result,
        date
    FROM read_csv_auto('{csv}', ignore_errors=true)
    WHERE LOWER(league) LIKE '%msi%'
      AND year = {year}
      AND playoffs = 1
    ORDER BY date DESC
    LIMIT 20
    """
    
    result = con.execute(query).fetchdf()
    if len(result) > 0:
        print(f"\n{year} MSI - Últimos jogos:")
        # Pegar times únicos dos últimos jogos
        teams = result.groupby('teamname')['result'].sum().sort_values(ascending=False)
        for team, wins in teams.items():
            print(f"  {team}: {int(wins)} vitórias")
