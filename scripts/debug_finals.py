"""Debug finals detection issues."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")

# Read the series parquet
series = "data/silver/series.parquet"

print("=" * 70)
print("DEBUGANDO DETECCAO DE FINAIS")
print("=" * 70)

# Check Worlds 2016 (showing Draw)
print("\n1. Worlds 2016 - Ultimos dias:")
query = f"""
SELECT 
    match_date,
    team1,
    team2,
    series_format,
    team1_wins,
    team2_wins,
    score,
    series_winner,
    tournament_phase,
    days_from_end
FROM read_parquet('{series}')
WHERE league = 'WLDs' AND year = 2016
ORDER BY match_date DESC
LIMIT 10
"""
result = con.execute(query).fetchdf()
print(result.to_string())

# Check Worlds 2017 (missing final)
print("\n\n2. Worlds 2017 - Ultimos dias:")
query = f"""
SELECT 
    match_date,
    team1,
    team2,
    series_format,
    team1_wins,
    team2_wins,
    score,
    series_winner,
    tournament_phase,
    days_from_end
FROM read_parquet('{series}')
WHERE league = 'WLDs' AND year = 2017
ORDER BY match_date DESC
LIMIT 10
"""
result = con.execute(query).fetchdf()
print(result.to_string())

# Check Worlds 2023 (missing final - T1 vs Weibo)
print("\n\n3. Worlds 2023 - Ultimos dias:")
query = f"""
SELECT 
    match_date,
    team1,
    team2,
    series_format,
    team1_wins,
    team2_wins,
    score,
    series_winner,
    tournament_phase,
    days_from_end
FROM read_parquet('{series}')
WHERE league = 'WLDs' AND year = 2023
ORDER BY match_date DESC
LIMIT 10
"""
result = con.execute(query).fetchdf()
print(result.to_string())

# Check if Samsung Galaxy vs SKT exists in 2017
print("\n\n4. Buscando Samsung Galaxy em 2017:")
query = f"""
SELECT 
    match_date,
    team1,
    team2,
    series_format,
    score,
    series_winner,
    tournament_phase
FROM read_parquet('{series}')
WHERE league = 'WLDs' AND year = 2017 
  AND (team1 LIKE '%Samsung%' OR team2 LIKE '%Samsung%' OR team1 LIKE '%SKT%' OR team2 LIKE '%SKT%')
ORDER BY match_date
"""
result = con.execute(query).fetchdf()
print(result.to_string())

# Check all Bo5 in Worlds 2017
print("\n\n5. Todos os Bo5 em Worlds 2017:")
query = f"""
SELECT 
    match_date,
    team1,
    team2,
    games_played,
    series_format,
    score,
    series_winner,
    tournament_phase
FROM read_parquet('{series}')
WHERE league = 'WLDs' AND year = 2017 AND series_format = 'Bo5'
ORDER BY match_date
"""
result = con.execute(query).fetchdf()
print(result.to_string())
