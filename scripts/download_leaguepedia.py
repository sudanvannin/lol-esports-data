"""
Download historical match data from Leaguepedia (Fandom wiki).
They have complete data going back to 2011.
"""

import httpx
import json
import time
from pathlib import Path
from datetime import datetime

BRONZE_PATH = Path("data/bronze/leaguepedia")
BRONZE_PATH.mkdir(parents=True, exist_ok=True)

# Leaguepedia Cargo API
API_URL = "https://lol.fandom.com/api.php"

def query_leaguepedia(tables: str, fields: str, where: str = "", limit: int = 500, offset: int = 0) -> list:
    """Query Leaguepedia Cargo API."""
    params = {
        "action": "cargoquery",
        "format": "json",
        "tables": tables,
        "fields": fields,
        "limit": limit,
        "offset": offset,
    }
    if where:
        params["where"] = where
    
    response = httpx.get(API_URL, params=params, timeout=30)
    data = response.json()
    
    results = []
    for item in data.get("cargoquery", []):
        results.append(item.get("title", {}))
    
    return results


def get_all_results(tables: str, fields: str, where: str = "") -> list:
    """Get all results with pagination."""
    all_results = []
    offset = 0
    limit = 500
    
    while True:
        results = query_leaguepedia(tables, fields, where, limit, offset)
        if not results:
            break
        all_results.extend(results)
        offset += limit
        
        if len(results) < limit:
            break
        
        time.sleep(0.5)  # Rate limiting
        
        if offset % 5000 == 0:
            print(f"    Fetched {offset} records...")
    
    return all_results


def download_player_stats():
    """Download player statistics for all games."""
    print("\n[1] Downloading player game statistics...")
    
    # ScoreboardPlayer has individual player stats per game
    fields = "Name,Link,Team,Champion,Role,Kills,Deaths,Assists,CS,Gold,DamageToChampions,VisionScore,DateTime UTC,GameId,MatchId,Tournament,Side,PlayerWin"
    
    # Get data year by year
    all_stats = []
    
    for year in range(2014, 2027):
        print(f"  Fetching {year}...")
        where = f'"DateTime UTC" >= "{year}-01-01" AND "DateTime UTC" < "{year + 1}-01-01"'
        stats = get_all_results("ScoreboardPlayer", fields, where)
        all_stats.extend(stats)
        print(f"    {year}: {len(stats)} player-game records")
    
    # Save to file
    output_path = BRONZE_PATH / "player_stats.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    
    print(f"  Saved {len(all_stats)} total records to {output_path}")
    return all_stats


def download_match_results():
    """Download match results."""
    print("\n[2] Downloading match results...")
    
    # Note: Field names have spaces in Leaguepedia
    fields = "MatchId,DateTime UTC,Team1,Team2,Winner,Team1Score,Team2Score,Tournament,MatchDay,BestOf"
    
    all_matches = []
    
    for year in range(2011, 2027):
        print(f"  Fetching {year}...")
        where = f'"DateTime UTC" >= "{year}-01-01" AND "DateTime UTC" < "{year + 1}-01-01"'
        matches = get_all_results("MatchSchedule", fields, where)
        all_matches.extend(matches)
        print(f"    {year}: {len(matches)} matches")
    
    output_path = BRONZE_PATH / "match_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_matches, f, ensure_ascii=False, indent=2)
    
    print(f"  Saved {len(all_matches)} total records to {output_path}")
    return all_matches


def download_player_info():
    """Download player information."""
    print("\n[3] Downloading player profiles...")
    
    fields = ",".join([
        "ID",
        "Player",
        "Name",
        "NationalityPrimary",
        "Birthdate",
        "Team",
        "Role",
        "IsRetired",
    ])
    
    players = get_all_results("Players", fields)
    
    output_path = BRONZE_PATH / "players.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(players, f, ensure_ascii=False, indent=2)
    
    print(f"  Saved {len(players)} players to {output_path}")
    return players


def main():
    start_time = datetime.now()
    print("=" * 70)
    print("LEAGUEPEDIA DATA DOWNLOAD")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Download all datasets
    matches = download_match_results()
    players = download_player_info()
    player_stats = download_player_stats()  # Individual player stats per game
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"Output: {BRONZE_PATH}")
    
    # Summary
    print("\nFiles created:")
    for f in sorted(BRONZE_PATH.glob("*.json")):
        size = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()
