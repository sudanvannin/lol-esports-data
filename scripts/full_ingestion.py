"""
Full historical ingestion - all matches + game details.
This will take a while but collects everything available.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.esports_api import LoLEsportsClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("full_ingestion")

BRONZE_PATH = Path("data/bronze")

LEAGUES_TO_COLLECT = [
    ("98767975604431411", "worlds"),
    ("98767991325878492", "msi"),
    ("98767991310872058", "lck"),
    ("98767991302996019", "lec"),
    ("98767991299243165", "lcs"),
    ("98767991314006698", "lpl"),
    ("98767991332355509", "cblol"),
]


def save_json(data: dict, data_type: str, identifier: str, date: datetime | None = None):
    """Save JSON data to bronze layer."""
    if date:
        path = BRONZE_PATH / data_type / date.strftime("%Y/%m/%d") / f"{identifier}.json"
    else:
        path = BRONZE_PATH / data_type / f"{identifier}.json"
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    envelope = {
        "ingested_at": datetime.utcnow().isoformat(),
        "source": "lol_esports_api",
        "data_type": data_type,
        "content": data,
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, ensure_ascii=False, indent=2, default=str)
    
    return str(path)


async def collect_all_matches(client: LoLEsportsClient, league_id: str, league_name: str) -> list[dict]:
    """Collect ALL matches for a league with full pagination."""
    matches = []
    page_token = None
    pages = 0
    max_pages = 500
    
    logger.info(f"Collecting matches for {league_name}...")
    
    while pages < max_pages:
        try:
            data = await client.get_schedule(league_id=league_id, page_token=page_token)
            schedule = data.get("schedule", {})
            events = schedule.get("events", [])
            pages += 1
            
            for event in events:
                if event.get("state") != "completed":
                    continue
                if event.get("type") != "match":
                    continue
                
                match_data = event.get("match", {})
                if not match_data:
                    continue
                
                match = {
                    "match_id": match_data.get("id"),
                    "state": event.get("state"),
                    "start_time": event.get("startTime"),
                    "block_name": event.get("blockName"),
                    "league": event.get("league", {}),
                    "tournament": event.get("tournament", {}),
                    "strategy": match_data.get("strategy", {}),
                    "teams": match_data.get("teams", []),
                    "games": match_data.get("games", []),
                }
                matches.append(match)
            
            # Pagination
            pagination = schedule.get("pages", {})
            page_token = pagination.get("older")
            
            if not page_token:
                break
            
            if pages % 20 == 0:
                logger.info(f"  {league_name}: {pages} pages, {len(matches)} matches...")
                
        except Exception as e:
            logger.error(f"Error on page {pages}: {e}")
            break
    
    logger.info(f"  {league_name}: DONE - {len(matches)} matches total")
    return matches


async def collect_game_details(client: LoLEsportsClient, match_id: str, game_ids: list[str]) -> list[dict]:
    """Collect detailed game data for a match."""
    games = []
    
    for game_id in game_ids:
        try:
            # Try to get window data (has player stats frame by frame)
            window_data = await client.get_game_window(game_id)
            if window_data:
                games.append({
                    "game_id": game_id,
                    "match_id": match_id,
                    "type": "window",
                    "data": window_data,
                })
        except Exception as e:
            logger.debug(f"No window data for game {game_id}: {e}")
    
    return games


async def main():
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("FULL HISTORICAL INGESTION")
    logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    all_matches = []
    
    async with LoLEsportsClient() as client:
        # Phase 1: Collect all matches
        logger.info("\n[PHASE 1] Collecting all matches...")
        
        for league_id, league_name in LEAGUES_TO_COLLECT:
            matches = await collect_all_matches(client, league_id, league_name)
            
            # Save each match
            for match in matches:
                try:
                    match_date = None
                    if match.get("start_time"):
                        match_date = datetime.fromisoformat(
                            match["start_time"].replace("Z", "+00:00")
                        )
                    
                    save_json(match, "matches", match["match_id"], match_date)
                    all_matches.append(match)
                except Exception as e:
                    logger.error(f"Error saving match {match.get('match_id')}: {e}")
        
        logger.info(f"\nTotal matches collected: {len(all_matches)}")
        
        # Phase 2: Collect game details (optional - takes longer)
        logger.info("\n[PHASE 2] Collecting game details...")
        
        games_collected = 0
        total_games = sum(len(m.get("games", [])) for m in all_matches)
        logger.info(f"Total games to process: {total_games}")
        
        for i, match in enumerate(all_matches):
            game_ids = [g.get("id") for g in match.get("games", []) if g.get("id")]
            
            if not game_ids:
                continue
            
            match_id = match["match_id"]
            
            try:
                match_date = None
                if match.get("start_time"):
                    match_date = datetime.fromisoformat(
                        match["start_time"].replace("Z", "+00:00")
                    )
                
                # Get event details (has more info about games)
                event_details = await client.get_event_details(match_id)
                if event_details:
                    save_json(event_details, "event_details", match_id, match_date)
                    games_collected += len(game_ids)
                
            except Exception as e:
                logger.debug(f"No event details for match {match_id}: {e}")
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i + 1}/{len(all_matches)} matches processed")
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "=" * 70)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Matches: {len(all_matches)}")
    logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
    logger.info(f"Output: {BRONZE_PATH}")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
