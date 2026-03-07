"""
Download ALL match data from Leaguepedia.
Since ScoreboardPlayer is empty, we'll use MatchSchedule which works.
"""

import httpx
import json
import time
from pathlib import Path
from datetime import datetime

BRONZE_PATH = Path("data/bronze/leaguepedia")
BRONZE_PATH.mkdir(parents=True, exist_ok=True)

API_URL = "https://lol.fandom.com/api.php"


def query_api(tables: str, fields: str, where: str = "", order_by: str = "", limit: int = 500, offset: int = 0) -> list:
    """Query Leaguepedia API."""
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
    if order_by:
        params["order_by"] = order_by
    
    try:
        response = httpx.get(API_URL, params=params, timeout=60)
        data = response.json()
        
        if "error" in data:
            print(f"API Error: {data['error']}")
            return []
        
        results = []
        for item in data.get("cargoquery", []):
            results.append(item.get("title", {}))
        
        return results
    except Exception as e:
        print(f"Request error: {e}")
        return []


def get_all_matches():
    """Download all matches from MatchSchedule."""
    print("Downloading all matches...")
    
    all_matches = []
    offset = 0
    limit = 500
    
    while True:
        results = query_api(
            tables="MatchSchedule",
            fields="MatchId,Team1,Team2,Winner,Team1Score,Team2Score,DateTime UTC,BestOf,Tournament,MatchDay",
            order_by="DateTime UTC DESC",
            limit=limit,
            offset=offset,
        )
        
        if not results:
            break
        
        all_matches.extend(results)
        offset += limit
        
        print(f"  Fetched {len(all_matches)} matches...")
        
        if len(results) < limit:
            break
        
        time.sleep(0.3)  # Rate limiting
    
    print(f"Total matches: {len(all_matches)}")
    return all_matches


def get_all_players():
    """Download all player profiles."""
    print("\nDownloading player profiles...")
    
    all_players = []
    offset = 0
    limit = 500
    
    while True:
        results = query_api(
            tables="Players",
            fields="ID,Player,Name,NationalityPrimary,Birthdate,Team,Role,IsRetired,Residency",
            limit=limit,
            offset=offset,
        )
        
        if not results:
            break
        
        all_players.extend(results)
        offset += limit
        
        if len(results) < limit:
            break
        
        time.sleep(0.3)
    
    print(f"Total players: {len(all_players)}")
    return all_players


def get_team_history(team_name: str) -> list:
    """Get match history for a specific team."""
    matches = []
    offset = 0
    
    while True:
        results = query_api(
            tables="MatchSchedule",
            fields="MatchId,Team1,Team2,Winner,Team1Score,Team2Score,DateTime UTC,Tournament",
            where=f'Team1="{team_name}" OR Team2="{team_name}"',
            order_by="DateTime UTC DESC",
            limit=500,
            offset=offset,
        )
        
        if not results:
            break
        
        matches.extend(results)
        offset += 500
        
        if len(results) < 500:
            break
        
        time.sleep(0.3)
    
    return matches


def calculate_team_winrate(matches: list, team_name: str) -> dict:
    """Calculate win rate for a team from match data."""
    wins = 0
    losses = 0
    
    for m in matches:
        team1 = m.get("Team1", "")
        team2 = m.get("Team2", "")
        winner = m.get("Winner", "")
        
        if team1 == team_name:
            if winner == "1":
                wins += 1
            elif winner == "2":
                losses += 1
        elif team2 == team_name:
            if winner == "2":
                wins += 1
            elif winner == "1":
                losses += 1
    
    total = wins + losses
    return {
        "team": team_name,
        "wins": wins,
        "losses": losses,
        "total": total,
        "winrate": (wins / total * 100) if total > 0 else 0,
    }


def main():
    start_time = datetime.now()
    print("=" * 70)
    print("LEAGUEPEDIA FULL DATA DOWNLOAD")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Download matches
    matches = get_all_matches()
    
    # Save matches
    output_path = BRONZE_PATH / "all_matches.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")
    
    # Download players
    players = get_all_players()
    
    output_path = BRONZE_PATH / "all_players.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(players, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")
    
    # Calculate stats for Gen.G (Chovy's team)
    print("\n" + "=" * 70)
    print("CALCULATING GEN.G STATS")
    print("=" * 70)
    
    geng_matches = get_team_history("Gen.G")
    print(f"Gen.G matches: {len(geng_matches)}")
    
    stats = calculate_team_winrate(geng_matches, "Gen.G")
    print(f"  Wins: {stats['wins']}")
    print(f"  Losses: {stats['losses']}")
    print(f"  Win Rate: {stats['winrate']:.1f}%")
    
    # Date range
    if geng_matches:
        dates = [m.get("DateTime UTC", "") for m in geng_matches if m.get("DateTime UTC")]
        if dates:
            print(f"  Date Range: {min(dates)[:10]} to {max(dates)[:10]}")
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Total matches: {len(matches)}")
    print(f"Total players: {len(players)}")
    print(f"Duration: {duration:.1f}s")


if __name__ == "__main__":
    main()
