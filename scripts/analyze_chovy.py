"""
Analyze Chovy's career stats from Oracle's Elixir data.
"""

import duckdb
from pathlib import Path

# Connect to DuckDB
con = duckdb.connect(":memory:")

print("=" * 70)
print("CHOVY CAREER ANALYSIS - Oracle's Elixir Data")
print("=" * 70)

# Load all CSV files
csv_files = list(Path("data/bronze/oracle_elixir").glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
print(f"\nFound {len(csv_files)} CSV files")

# Check columns first
print("\nChecking data structure...")
sample = con.execute(f"SELECT * FROM read_csv_auto('{csv_files[0]}') LIMIT 1").fetchdf()
print(f"Columns: {list(sample.columns)}")

# Create a view combining all CSVs
csv_pattern = "data/bronze/oracle_elixir/*_LoL_esports_match_data_from_OraclesElixir.csv"

print("\n" + "=" * 70)
print("CHOVY CAREER STATS")
print("=" * 70)

# Find Chovy's games
query = f"""
SELECT 
    playername,
    COUNT(*) as games,
    SUM(CASE WHEN result = 1 THEN 1 ELSE 0 END) as wins,
    ROUND(AVG(kills), 2) as avg_kills,
    ROUND(AVG(deaths), 2) as avg_deaths,
    ROUND(AVG(assists), 2) as avg_assists,
    ROUND(AVG(CASE WHEN deaths > 0 THEN (kills + assists) / deaths ELSE kills + assists END), 2) as kda,
    ROUND(AVG(dpm), 0) as avg_dpm,
    ROUND(AVG(cspm), 1) as avg_cspm,
    ROUND(AVG(golddiffat15), 0) as avg_gd15
FROM read_csv_auto('{csv_pattern}', ignore_errors=true)
WHERE LOWER(playername) LIKE '%chovy%'
GROUP BY playername
"""

result = con.execute(query).fetchdf()
print("\nChovy found in data:")
print(result.to_string(index=False))

if len(result) > 0:
    total_games = result['games'].sum()
    total_wins = result['wins'].sum()
    winrate = (total_wins / total_games * 100) if total_games > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"CHOVY OVERALL WIN RATE: {winrate:.1f}%")
    print(f"Total Games: {total_games}")
    print(f"Wins: {total_wins} | Losses: {total_games - total_wins}")
    print(f"{'='*70}")

# By year
print("\n" + "=" * 70)
print("CHOVY WIN RATE BY YEAR")
print("=" * 70)

query = f"""
SELECT 
    year,
    teamname as team,
    COUNT(*) as games,
    SUM(CASE WHEN result = 1 THEN 1 ELSE 0 END) as wins,
    ROUND(100.0 * SUM(CASE WHEN result = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as winrate,
    ROUND(AVG(kills), 1) as avg_k,
    ROUND(AVG(deaths), 1) as avg_d,
    ROUND(AVG(assists), 1) as avg_a
FROM read_csv_auto('{csv_pattern}', ignore_errors=true)
WHERE LOWER(playername) LIKE '%chovy%'
GROUP BY year, teamname
ORDER BY year, games DESC
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))

# By champion
print("\n" + "=" * 70)
print("CHOVY TOP CHAMPIONS (min 10 games)")
print("=" * 70)

query = f"""
SELECT 
    champion,
    COUNT(*) as games,
    SUM(CASE WHEN result = 1 THEN 1 ELSE 0 END) as wins,
    ROUND(100.0 * SUM(CASE WHEN result = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as winrate,
    ROUND(AVG(kills), 1) as avg_k,
    ROUND(AVG(deaths), 1) as avg_d,
    ROUND(AVG(assists), 1) as avg_a
FROM read_csv_auto('{csv_pattern}', ignore_errors=true)
WHERE LOWER(playername) LIKE '%chovy%'
GROUP BY champion
HAVING COUNT(*) >= 10
ORDER BY games DESC
LIMIT 15
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))

# Vs top teams
print("\n" + "=" * 70)
print("CHOVY VS TOP TEAMS")
print("=" * 70)

query = f"""
WITH chovy_games AS (
    SELECT 
        gameid,
        teamname as chovy_team,
        result
    FROM read_csv_auto('{csv_pattern}', ignore_errors=true)
    WHERE LOWER(playername) LIKE '%chovy%'
),
opponent_teams AS (
    SELECT DISTINCT
        g.gameid,
        g.teamname as opponent
    FROM read_csv_auto('{csv_pattern}', ignore_errors=true) g
    JOIN chovy_games c ON g.gameid = c.gameid
    WHERE g.teamname != c.chovy_team
)
SELECT 
    o.opponent,
    COUNT(*) as games,
    SUM(CASE WHEN c.result = 1 THEN 1 ELSE 0 END) as wins,
    ROUND(100.0 * SUM(CASE WHEN c.result = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as winrate
FROM chovy_games c
JOIN opponent_teams o ON c.gameid = o.gameid
GROUP BY o.opponent
HAVING COUNT(*) >= 5
ORDER BY games DESC
LIMIT 15
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))

# International performance
print("\n" + "=" * 70)
print("CHOVY INTERNATIONAL (Worlds, MSI)")
print("=" * 70)

query = f"""
SELECT 
    league,
    COUNT(*) as games,
    SUM(CASE WHEN result = 1 THEN 1 ELSE 0 END) as wins,
    ROUND(100.0 * SUM(CASE WHEN result = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as winrate
FROM read_csv_auto('{csv_pattern}', ignore_errors=true)
WHERE LOWER(playername) LIKE '%chovy%'
  AND (LOWER(league) LIKE '%worlds%' OR LOWER(league) LIKE '%msi%' OR LOWER(league) LIKE '%international%')
GROUP BY league
ORDER BY games DESC
"""

result = con.execute(query).fetchdf()
if len(result) > 0:
    print(result.to_string(index=False))
else:
    print("No international data found with current filters")

print("\n" + "=" * 70)
print("DATA RANGE")
print("=" * 70)

query = f"""
SELECT 
    MIN(date::DATE) as first_game,
    MAX(date::DATE) as last_game,
    COUNT(DISTINCT gameid) as total_unique_games
FROM read_csv_auto('{csv_pattern}', ignore_errors=true)
WHERE LOWER(playername) LIKE '%chovy%'
"""

result = con.execute(query).fetchdf()
print(f"First game: {result.iloc[0, 0]}")
print(f"Last game: {result.iloc[0, 1]}")
print(f"Total unique games: {result.iloc[0, 2]}")
