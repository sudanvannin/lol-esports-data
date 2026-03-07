"""
Chovy vs Faker - Head to Head Analysis
"""

import duckdb

con = duckdb.connect(":memory:")

csv_pattern = "data/bronze/oracle_elixir/*_LoL_esports_match_data_from_OraclesElixir.csv"

print("=" * 70)
print("CHOVY vs FAKER - ANÁLISE COMPLETA")
print("=" * 70)

# Career stats comparison
print("\n[1] ESTATÍSTICAS DE CARREIRA")
print("-" * 70)

query = f"""
SELECT 
    playername,
    COUNT(*) as games,
    SUM(CASE WHEN result = 1 THEN 1 ELSE 0 END) as wins,
    ROUND(100.0 * SUM(CASE WHEN result = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as winrate,
    ROUND(AVG(kills), 2) as avg_kills,
    ROUND(AVG(deaths), 2) as avg_deaths,
    ROUND(AVG(assists), 2) as avg_assists,
    ROUND(AVG(CASE WHEN deaths > 0 THEN (kills + assists) / deaths ELSE kills + assists END), 2) as kda,
    ROUND(AVG(dpm), 0) as dpm,
    ROUND(AVG(cspm), 1) as cspm,
    ROUND(AVG(golddiffat15), 0) as gd15
FROM read_csv_auto('{csv_pattern}', ignore_errors=true)
WHERE LOWER(playername) IN ('chovy', 'faker')
GROUP BY playername
ORDER BY games DESC
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))

# Year by year comparison
print("\n\n[2] EVOLUÇÃO POR ANO")
print("-" * 70)

query = f"""
SELECT 
    year,
    playername,
    teamname as team,
    COUNT(*) as games,
    ROUND(100.0 * SUM(CASE WHEN result = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as winrate,
    ROUND(AVG(kills), 1) as k,
    ROUND(AVG(deaths), 1) as d,
    ROUND(AVG(assists), 1) as a
FROM read_csv_auto('{csv_pattern}', ignore_errors=true)
WHERE LOWER(playername) IN ('chovy', 'faker')
GROUP BY year, playername, teamname
ORDER BY year, playername
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))

# Head to head - games where both played
print("\n\n[3] CONFRONTOS DIRETOS (Head to Head)")
print("-" * 70)

query = f"""
WITH chovy_games AS (
    SELECT gameid, teamname as chovy_team, result as chovy_result,
           kills as chovy_kills, deaths as chovy_deaths, assists as chovy_assists,
           champion as chovy_champ, dpm as chovy_dpm, golddiffat15 as chovy_gd15
    FROM read_csv_auto('{csv_pattern}', ignore_errors=true)
    WHERE LOWER(playername) = 'chovy'
),
faker_games AS (
    SELECT gameid, teamname as faker_team, result as faker_result,
           kills as faker_kills, deaths as faker_deaths, assists as faker_assists,
           champion as faker_champ, dpm as faker_dpm, golddiffat15 as faker_gd15
    FROM read_csv_auto('{csv_pattern}', ignore_errors=true)
    WHERE LOWER(playername) = 'faker'
)
SELECT 
    COUNT(*) as total_games,
    SUM(CASE WHEN c.chovy_result = 1 THEN 1 ELSE 0 END) as chovy_wins,
    SUM(CASE WHEN f.faker_result = 1 THEN 1 ELSE 0 END) as faker_wins,
    ROUND(AVG(c.chovy_kills), 1) as chovy_avg_k,
    ROUND(AVG(c.chovy_deaths), 1) as chovy_avg_d,
    ROUND(AVG(c.chovy_assists), 1) as chovy_avg_a,
    ROUND(AVG(f.faker_kills), 1) as faker_avg_k,
    ROUND(AVG(f.faker_deaths), 1) as faker_avg_d,
    ROUND(AVG(f.faker_assists), 1) as faker_avg_a
FROM chovy_games c
JOIN faker_games f ON c.gameid = f.gameid
WHERE c.chovy_team != f.faker_team
"""

result = con.execute(query).fetchdf()
if len(result) > 0 and result.iloc[0, 0] > 0:
    total = result.iloc[0, 0]
    chovy_wins = result.iloc[0, 1]
    faker_wins = result.iloc[0, 2]
    
    print(f"Total de jogos diretos: {total}")
    print(f"\nPlacar:")
    print(f"  Chovy: {int(chovy_wins)} vitórias ({chovy_wins/total*100:.1f}%)")
    print(f"  Faker: {int(faker_wins)} vitórias ({faker_wins/total*100:.1f}%)")
    print(f"\nMédias nos confrontos diretos:")
    print(f"  Chovy: {result.iloc[0, 3]}/{result.iloc[0, 4]}/{result.iloc[0, 5]} KDA")
    print(f"  Faker: {result.iloc[0, 6]}/{result.iloc[0, 7]}/{result.iloc[0, 8]} KDA")

# Head to head by year
print("\n\n[4] CONFRONTOS DIRETOS POR ANO")
print("-" * 70)

query = f"""
WITH chovy_games AS (
    SELECT gameid, year, teamname as chovy_team, result as chovy_result
    FROM read_csv_auto('{csv_pattern}', ignore_errors=true)
    WHERE LOWER(playername) = 'chovy'
),
faker_games AS (
    SELECT gameid, teamname as faker_team, result as faker_result
    FROM read_csv_auto('{csv_pattern}', ignore_errors=true)
    WHERE LOWER(playername) = 'faker'
)
SELECT 
    c.year,
    c.chovy_team,
    f.faker_team,
    COUNT(*) as games,
    SUM(CASE WHEN c.chovy_result = 1 THEN 1 ELSE 0 END) as chovy_wins,
    SUM(CASE WHEN f.faker_result = 1 THEN 1 ELSE 0 END) as faker_wins
FROM chovy_games c
JOIN faker_games f ON c.gameid = f.gameid
WHERE c.chovy_team != f.faker_team
GROUP BY c.year, c.chovy_team, f.faker_team
ORDER BY c.year
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))

# Champion matchups
print("\n\n[5] MATCHUPS DE CHAMPIONS (min 3 jogos)")
print("-" * 70)

query = f"""
WITH chovy_games AS (
    SELECT gameid, champion as chovy_champ, result as chovy_result
    FROM read_csv_auto('{csv_pattern}', ignore_errors=true)
    WHERE LOWER(playername) = 'chovy'
),
faker_games AS (
    SELECT gameid, champion as faker_champ, result as faker_result
    FROM read_csv_auto('{csv_pattern}', ignore_errors=true)
    WHERE LOWER(playername) = 'faker'
)
SELECT 
    c.chovy_champ as Chovy,
    f.faker_champ as Faker,
    COUNT(*) as games,
    SUM(CASE WHEN c.chovy_result = 1 THEN 1 ELSE 0 END) as chovy_wins,
    SUM(CASE WHEN f.faker_result = 1 THEN 1 ELSE 0 END) as faker_wins
FROM chovy_games c
JOIN faker_games f ON c.gameid = f.gameid
WHERE c.chovy_result != f.faker_result  -- Same game, different teams
GROUP BY c.chovy_champ, f.faker_champ
HAVING COUNT(*) >= 3
ORDER BY games DESC
LIMIT 15
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))

# Titles comparison
print("\n\n[6] TÍTULOS E CONQUISTAS")
print("-" * 70)

query = f"""
WITH player_finals AS (
    SELECT 
        playername,
        league,
        split,
        year,
        MAX(result) as won_final
    FROM read_csv_auto('{csv_pattern}', ignore_errors=true)
    WHERE LOWER(playername) IN ('chovy', 'faker')
      AND playoffs = 1
    GROUP BY playername, league, split, year
)
SELECT 
    playername,
    COUNT(*) as playoff_appearances,
    SUM(won_final) as titles_approx
FROM player_finals
GROUP BY playername
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))

# International
print("\n\n[7] PERFORMANCE INTERNACIONAL")
print("-" * 70)

query = f"""
SELECT 
    playername,
    league,
    COUNT(*) as games,
    SUM(CASE WHEN result = 1 THEN 1 ELSE 0 END) as wins,
    ROUND(100.0 * SUM(CASE WHEN result = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as winrate
FROM read_csv_auto('{csv_pattern}', ignore_errors=true)
WHERE LOWER(playername) IN ('chovy', 'faker')
  AND (LOWER(league) LIKE '%worlds%' OR LOWER(league) LIKE '%msi%')
GROUP BY playername, league
ORDER BY playername, games DESC
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))

print("\n" + "=" * 70)
print("FIM DA ANÁLISE")
print("=" * 70)
