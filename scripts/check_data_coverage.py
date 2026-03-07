"""Check data coverage and quality."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")
csv = "data/bronze/oracle_elixir/*_LoL_esports_match_data_from_OraclesElixir.csv"

print("=" * 70)
print("VERIFICANDO COBERTURA DOS DADOS")
print("=" * 70)

# Anos disponíveis
print("\n1. Anos disponíveis:")
query = f"""
SELECT DISTINCT year
FROM read_csv_auto('{csv}', ignore_errors=true)
ORDER BY year
"""
result = con.execute(query).fetchdf()
years = [int(y) for y in result['year'].tolist()]
print(f"   {years}")
print(f"   Faltando: 2025" if 2025 not in years else "   2025 presente")

# Worlds por ano
print("\n2. Dados de Worlds (WLDs) por ano:")
query = f"""
SELECT 
    year,
    COUNT(DISTINCT gameid) as games,
    COUNT(DISTINCT teamname) as teams
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE league = 'WLDs'
GROUP BY year
ORDER BY year
"""
result = con.execute(query).fetchdf()
for _, row in result.iterrows():
    print(f"   {int(row['year'])}: {int(row['games'])} jogos, {int(row['teams'])} times")

# Faker em Worlds
print("\n3. Faker em Worlds por ano:")
query = f"""
SELECT 
    year,
    teamname,
    COUNT(*) as games,
    SUM(result) as wins
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE league = 'WLDs' AND LOWER(playername) = 'faker'
GROUP BY year, teamname
ORDER BY year
"""
result = con.execute(query).fetchdf()
for _, row in result.iterrows():
    wins = int(row['wins'])
    games = int(row['games'])
    print(f"   {int(row['year'])} ({row['teamname']}): {wins}W-{games-wins}L")

# Verificar campeões de Worlds (time com mais vitórias em cada ano)
print("\n4. Times com mais vitórias em Worlds (provável campeão):")
query = f"""
WITH team_wins AS (
    SELECT 
        year,
        teamname,
        SUM(result) as wins,
        COUNT(*) as games
    FROM read_csv_auto('{csv}', ignore_errors=true)
    WHERE league = 'WLDs'
    GROUP BY year, teamname
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY year ORDER BY wins DESC) as rn
    FROM team_wins
)
SELECT year, teamname, wins, games
FROM ranked
WHERE rn = 1
ORDER BY year
"""
result = con.execute(query).fetchdf()
for _, row in result.iterrows():
    wins = int(row['wins'])
    games = int(row['games'])
    print(f"   {int(row['year'])}: {row['teamname']} ({wins}W)")

# MSI por ano
print("\n5. Dados de MSI por ano:")
query = f"""
SELECT 
    year,
    COUNT(DISTINCT gameid) as games,
    COUNT(DISTINCT teamname) as teams
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE league = 'MSI'
GROUP BY year
ORDER BY year
"""
result = con.execute(query).fetchdf()
for _, row in result.iterrows():
    print(f"   {int(row['year'])}: {int(row['games'])} jogos, {int(row['teams'])} times")
