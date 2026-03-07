"""Check tournament phase indicators in data."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")
csv = "data/bronze/oracle_elixir/*_LoL_esports_match_data_from_OraclesElixir.csv"

print("=" * 70)
print("ANALISE DE FASES DE TORNEIO")
print("=" * 70)

# 1. Valores do campo playoffs
print("\n1. Campo 'playoffs' - valores unicos:")
query = f"""
SELECT DISTINCT playoffs, COUNT(*) as games
FROM read_csv_auto('{csv}', ignore_errors=true)
GROUP BY playoffs
ORDER BY games DESC
"""
result = con.execute(query).fetchdf()
for _, row in result.iterrows():
    print(f"   {row['playoffs']}: {row['games']} jogos")

# 2. Verificar se há campo de fase/stage
print("\n2. Exemplo de dados para identificar padroes:")
query = f"""
SELECT DISTINCT 
    league,
    split,
    playoffs,
    COUNT(DISTINCT gameid) as games
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE league IN ('WLDs', 'MSI', 'LCK')
GROUP BY league, split, playoffs
ORDER BY league, split
LIMIT 30
"""
result = con.execute(query).fetchdf()
print(result.to_string())

# 3. Worlds - padroes de jogos
print("\n\n3. Worlds - analise por ano e playoffs:")
query = f"""
SELECT 
    year,
    playoffs,
    COUNT(DISTINCT gameid) as games,
    COUNT(DISTINCT teamname) as teams
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE league = 'WLDs'
GROUP BY year, playoffs
ORDER BY year, playoffs
"""
result = con.execute(query).fetchdf()
print(result.to_string())

# 4. Tentar identificar finais pelo numero de jogos entre times
print("\n\n4. Worlds 2024 - Partidas (agrupadas por times):")
query = f"""
SELECT 
    LEAST(t1.teamname, t2.teamname) as team1,
    GREATEST(t1.teamname, t2.teamname) as team2,
    COUNT(DISTINCT t1.gameid) as games,
    SUM(CASE WHEN t1.result = 1 THEN 1 ELSE 0 END) as team1_wins
FROM read_csv_auto('{csv}', ignore_errors=true) t1
JOIN read_csv_auto('{csv}', ignore_errors=true) t2 
    ON t1.gameid = t2.gameid AND t1.teamname != t2.teamname
WHERE t1.league = 'WLDs' AND t1.year = 2024 AND t1.playoffs = 1 AND t1.position = 'team'
GROUP BY LEAST(t1.teamname, t2.teamname), GREATEST(t1.teamname, t2.teamname)
ORDER BY games DESC
LIMIT 10
"""
result = con.execute(query).fetchdf()
print(result.to_string())

# 5. Verificar campo split
print("\n\n5. Valores do campo 'split':")
query = f"""
SELECT DISTINCT split, COUNT(*) as count
FROM read_csv_auto('{csv}', ignore_errors=true)
GROUP BY split
ORDER BY count DESC
LIMIT 20
"""
result = con.execute(query).fetchdf()
print(result.to_string())
