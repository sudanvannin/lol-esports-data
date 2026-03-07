"""Check new champions in data."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")

champs = ['Zaahen', 'Yunara', 'Mel', 'Ambessa', 'Aurora']

for champ in champs:
    query = f"""
    SELECT 
        MIN(game_date) as first_seen,
        MAX(game_date) as last_seen,
        COUNT(*) as games
    FROM read_parquet('data/silver/players/**/*.parquet')
    WHERE champion = '{champ}'
    """
    result = con.execute(query).fetchdf()
    first = result['first_seen'].iloc[0]
    last = result['last_seen'].iloc[0]
    games = result['games'].iloc[0]
    print(f"{champ}: {games} jogos ({first} a {last})")
