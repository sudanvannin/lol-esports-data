"""Query specific game details."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import duckdb

con = duckdb.connect(":memory:")
csv = "data/bronze/oracle_elixir/*_LoL_esports_match_data_from_OraclesElixir.csv"

print("=" * 70)
print("MSI 2025 FINAL - GEN.G vs T1")
print("=" * 70)

# Find the final games
query = f"""
SELECT 
    gameid,
    game,
    CAST(date AS DATE) as match_date,
    teamname,
    playername,
    position,
    champion,
    result
FROM read_csv_auto('{csv}', ignore_errors=true)
WHERE league = 'MSI' 
    AND year = 2025 
    AND (teamname = 'Gen.G' OR teamname = 'T1')
    AND position != 'team'
ORDER BY date DESC, game DESC, teamname, position
LIMIT 50
"""
result = con.execute(query).fetchdf()

# Filter for final (last date)
last_date = result['match_date'].max()
final_games = result[result['match_date'] == last_date]

print(f"\nData da final: {last_date}")
print(f"Jogos encontrados: {final_games['game'].nunique()}")

# Last game
last_game = final_games['game'].max()
print(f"\nUltimo jogo (Game {last_game}):")
print("-" * 70)

game_data = final_games[final_games['game'] == last_game]
for team in game_data['teamname'].unique():
    team_data = game_data[game_data['teamname'] == team]
    result_str = "WIN" if team_data['result'].iloc[0] == 1 else "LOSS"
    print(f"\n{team} ({result_str}):")
    for _, row in team_data.iterrows():
        print(f"  {row['position']}: {row['playername']} - {row['champion']}")

# Chovy specifically
print("\n" + "=" * 70)
print("CHOVY NA FINAL DO MSI 2025")
print("=" * 70)

chovy_games = final_games[final_games['playername'].str.lower() == 'chovy']
for _, row in chovy_games.sort_values('game').iterrows():
    result_str = "WIN" if row['result'] == 1 else "LOSS"
    print(f"Game {row['game']}: {row['champion']} ({result_str})")
