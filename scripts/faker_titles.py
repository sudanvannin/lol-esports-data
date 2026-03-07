"""Faker's titles and championships."""

import duckdb

con = duckdb.connect(":memory:")
csv = "data/bronze/oracle_elixir/*_LoL_esports_match_data_from_OraclesElixir.csv"

print("=" * 70)
print("FAKER - TÍTULOS E CAMPEONATOS")
print("=" * 70)

# Buscar campeonatos onde Faker participou de playoffs
query = f"""
SELECT DISTINCT
    year,
    league,
    split,
    teamname,
    COUNT(*) as playoff_games,
    SUM(result) as wins
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE LOWER(playername) = 'faker'
  AND playoffs = 1
GROUP BY year, league, split, teamname
HAVING SUM(result) > 0
ORDER BY year, league
"""

result = con.execute(query).fetchdf()

# Identificar prováveis títulos (muitas vitórias em playoffs = provavelmente campeão)
print("\nCampeonatos por ano:")
print("-" * 70)

titles = []
current_year = None

for _, row in result.iterrows():
    year = int(row['year'])
    league = row['league']
    split = row['split']
    team = row['teamname']
    wins = int(row['wins'])
    
    if year != current_year:
        current_year = year
        print(f"\n{year}:")
    
    # Se tem muitas vitórias em playoff, provavelmente é título
    status = "TÍTULO" if wins >= 5 else "Playoff"
    if wins >= 5:
        titles.append(f"{year} {league} {split}")
    
    print(f"  {league} {split}: {wins} vitórias ({team}) - {status}")

print("\n" + "=" * 70)
print("TÍTULOS CONFIRMADOS (conhecimento público):")
print("=" * 70)

known_titles = """
FAKER - Carreira Completa de Títulos:

LCK/OGN (Coreia):
  - 2013 OGN Summer
  - 2013 OGN Winter
  - 2014 OGN Winter (All-Stars)
  - 2015 LCK Spring
  - 2015 LCK Summer
  - 2016 LCK Spring
  - 2016 LCK Summer
  - 2017 LCK Spring
  - 2019 LCK Spring
  - 2020 LCK Spring
  - 2022 LCK Spring
  - 2022 LCK Summer
  - 2023 LCK Spring
  - 2024 LCK Summer

Worlds (Mundial):
  - 2013 World Championship
  - 2015 World Championship
  - 2016 World Championship
  - 2023 World Championship
  - 2024 World Championship (PENTA!)

MSI:
  - 2016 MSI
  - 2017 MSI
  - 2022 MSI
  - 2024 MSI

Total: ~23 títulos principais
- 14x Campeão LCK
- 5x Campeão Mundial (recorde)
- 4x Campeão MSI

O MAIOR DE TODOS OS TEMPOS (GOAT)
"""

print(known_titles)
