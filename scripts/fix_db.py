import os

file_path = "web/db.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace the read_parquet('VAR') patterns that lost their variables
content = content.replace("read_parquet('{GAMES_PATH}')", "games")
content = content.replace("read_parquet('{PLAYERS_PATH}')", "players")
content = content.replace("read_parquet('{SERIES_PATH}')", "series")
content = content.replace("read_parquet('{CHAMPIONS_PATH}')", "champions")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed web/db.py successfully.")
