"""Check LCK matches from last week using Riot API."""

import asyncio
from datetime import datetime, timedelta
import sys
sys.path.insert(0, ".")

from src.ingestion.esports_api import LoLEsportsClient


async def main():
    print("=" * 70)
    print("LCK - PARTIDAS DA ULTIMA SEMANA")
    print("=" * 70)
    
    async with LoLEsportsClient() as client:
        # LCK league ID
        lck_id = "98767991310872058"
        
        # Get schedule
        data = await client.get_schedule(league_id=lck_id)
        schedule = data.get("schedule", {})
        events = schedule.get("events", [])
        
        now = datetime.now()
        one_week_ago = now - timedelta(days=7)
        
        print(f"\nData atual: {now.strftime('%Y-%m-%d')}")
        print(f"Buscando desde: {one_week_ago.strftime('%Y-%m-%d')}")
        print("-" * 70)
        
        recent_matches = []
        
        for event in events:
            if event.get("type") != "match":
                continue
            
            start_time_str = event.get("startTime")
            if not start_time_str:
                continue
            
            try:
                start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                start_time = start_time.replace(tzinfo=None)
            except:
                continue
            
            # Partidas da ultima semana
            if start_time >= one_week_ago and start_time <= now:
                match_data = event.get("match", {})
                teams = match_data.get("teams", [])
                
                if len(teams) >= 2:
                    team1 = teams[0]
                    team2 = teams[1]
                    
                    # Determinar vencedor
                    winner = None
                    if team1.get("result", {}).get("outcome") == "win":
                        winner = team1.get("name")
                    elif team2.get("result", {}).get("outcome") == "win":
                        winner = team2.get("name")
                    
                    score1 = team1.get("result", {}).get("gameWins", 0)
                    score2 = team2.get("result", {}).get("gameWins", 0)
                    
                    recent_matches.append({
                        "date": start_time,
                        "team1": team1.get("name"),
                        "team2": team2.get("name"),
                        "score": f"{score1}-{score2}",
                        "winner": winner,
                        "state": event.get("state")
                    })
        
        if recent_matches:
            recent_matches.sort(key=lambda x: x["date"])
            
            print(f"\nEncontradas {len(recent_matches)} partidas:\n")
            for m in recent_matches:
                date_str = m["date"].strftime("%Y-%m-%d %H:%M")
                status = "FINALIZADA" if m["state"] == "completed" else m["state"].upper()
                winner_str = f" -> {m['winner']}" if m["winner"] else ""
                print(f"  {date_str} | {m['team1']} vs {m['team2']} ({m['score']}) [{status}]{winner_str}")
        else:
            print("\nNenhuma partida encontrada na ultima semana.")
            print("Verificando partidas mais recentes disponiveis...")
            
            # Mostrar as 10 partidas mais recentes
            all_matches = []
            for event in events:
                if event.get("type") != "match":
                    continue
                if event.get("state") != "completed":
                    continue
                    
                start_time_str = event.get("startTime")
                if not start_time_str:
                    continue
                
                try:
                    start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                    start_time = start_time.replace(tzinfo=None)
                except:
                    continue
                
                match_data = event.get("match", {})
                teams = match_data.get("teams", [])
                
                if len(teams) >= 2:
                    team1 = teams[0]
                    team2 = teams[1]
                    
                    winner = None
                    if team1.get("result", {}).get("outcome") == "win":
                        winner = team1.get("name")
                    elif team2.get("result", {}).get("outcome") == "win":
                        winner = team2.get("name")
                    
                    score1 = team1.get("result", {}).get("gameWins", 0)
                    score2 = team2.get("result", {}).get("gameWins", 0)
                    
                    all_matches.append({
                        "date": start_time,
                        "team1": team1.get("name"),
                        "team2": team2.get("name"),
                        "score": f"{score1}-{score2}",
                        "winner": winner
                    })
            
            all_matches.sort(key=lambda x: x["date"], reverse=True)
            
            print(f"\n10 partidas mais recentes da LCK:")
            for m in all_matches[:10]:
                date_str = m["date"].strftime("%Y-%m-%d %H:%M")
                winner_str = f" -> {m['winner']}" if m["winner"] else ""
                print(f"  {date_str} | {m['team1']} vs {m['team2']} ({m['score']}){winner_str}")


if __name__ == "__main__":
    asyncio.run(main())
