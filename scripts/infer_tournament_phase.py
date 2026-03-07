"""Try to infer tournament phases from game patterns."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")
csv = "data/bronze/oracle_elixir/*_LoL_esports_match_data_from_OraclesElixir.csv"

print("=" * 70)
print("INFERINDO FASES DE TORNEIO")
print("=" * 70)

# Worlds 2024 - agrupar jogos entre mesmos times
print("\n1. Worlds 2024 - Series entre times (inferindo Bo1/Bo3/Bo5):")
query = f"""
WITH team_games AS (
    SELECT 
        gameid,
        date,
        teamname,
        result,
        ROW_NUMBER() OVER (PARTITION BY gameid ORDER BY teamname) as team_num
    FROM read_csv_auto('{csv}', ignore_errors=true)
    WHERE league = 'WLDs' AND year = 2024 AND position = 'team'
),
matchups AS (
    SELECT 
        t1.gameid,
        t1.date,
        t1.teamname as team1,
        t2.teamname as team2,
        t1.result as team1_win
    FROM team_games t1
    JOIN team_games t2 ON t1.gameid = t2.gameid AND t1.team_num = 1 AND t2.team_num = 2
),
daily_matchups AS (
    SELECT 
        CAST(date AS DATE) as match_date,
        team1,
        team2,
        COUNT(*) as games,
        SUM(team1_win) as team1_wins,
        COUNT(*) - SUM(team1_win) as team2_wins
    FROM matchups
    GROUP BY CAST(date AS DATE), team1, team2
)
SELECT 
    match_date,
    team1,
    team2,
    games,
    team1_wins || '-' || team2_wins as score,
    CASE 
        WHEN games = 1 THEN 'Bo1 (Groups?)'
        WHEN games <= 3 THEN 'Bo3 (Swiss?)'
        WHEN games <= 5 THEN 'Bo5 (Playoffs?)'
        ELSE 'Unknown'
    END as inferred_format
FROM daily_matchups
ORDER BY match_date DESC
LIMIT 30
"""
result = con.execute(query).fetchdf()
print(result.to_string())

# Verificar final 2024 - T1 vs BLG
print("\n\n2. Worlds 2024 Final - T1 vs Bilibili Gaming:")
query = f"""
SELECT 
    CAST(date AS DATE) as match_date,
    gameid,
    teamname,
    result
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE league = 'WLDs' 
    AND year = 2024 
    AND position = 'team'
    AND (LOWER(teamname) LIKE '%bilibili%' OR teamname = 'T1')
ORDER BY date DESC
LIMIT 20
"""
result = con.execute(query).fetchdf()
print(result.to_string())

# Verificar campos adicionais que possam ajudar
print("\n\n3. Verificando campo 'game' (pode indicar Game 1, 2, 3...):")
query = f"""
SELECT DISTINCT game, COUNT(*) as count
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE league = 'WLDs' AND year = 2024
GROUP BY game
ORDER BY count DESC
LIMIT 10
"""
result = con.execute(query).fetchdf()
print(result.to_string())

# Tentar ver padrão temporal para identificar fases
print("\n\n4. Worlds 2024 - Distribuicao de jogos por data:")
query = f"""
SELECT 
    CAST(date AS DATE) as match_date,
    COUNT(DISTINCT gameid) as games,
    COUNT(DISTINCT teamname) as teams
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE league = 'WLDs' AND year = 2024 AND position = 'team'
GROUP BY CAST(date AS DATE)
ORDER BY match_date
"""
result = con.execute(query).fetchdf()
print(result.to_string())
