"""
Client for LoL Esports API.

This module provides async clients for fetching data from the official
LoL Esports API and live stats feed.

API Documentation (unofficial):
- Base URL: https://esports-api.lolesports.com/persisted/gw
- All requests require x-api-key header
- Rate limit: ~100 requests/minute

Endpoints:
- /getLeagues: List all leagues (LCK, LEC, CBLOL, etc.)
- /getTournamentsForLeague: Tournaments for a league
- /getSchedule: Match schedule
- /getEventDetails: Details of a specific event/match
- /getGames: Game details with stats
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings
from src.models.esports import League, Match, MatchState, Team, Tournament

logger = logging.getLogger(__name__)

API_BASE_URL = "https://esports-api.lolesports.com/persisted/gw"
FEED_BASE_URL = "https://feed.lolesports.com/livestats/v1"

DEFAULT_API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"

DEFAULT_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""

    pass


class LoLEsportsClient:
    """
    Async client for LoL Esports API.

    Usage:
        async with LoLEsportsClient() as client:
            leagues = await client.get_leagues()

    The client handles rate limiting, retries, and error handling automatically.
    Configuration is loaded from environment variables via Settings.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float | None = None,
        max_concurrent: int | None = None,
    ):
        """
        Initialize the client.

        Args:
            api_key: API key (defaults to settings or public key)
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent requests
        """
        settings = get_settings()

        self.api_key = api_key or settings.api.lol_esports_api_key
        self.timeout = timeout or settings.api.request_timeout
        self.max_concurrent = max_concurrent or settings.api.max_concurrent_requests
        self.max_retries = settings.api.max_retries

        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self._client: httpx.AsyncClient | None = None
        self._request_count = 0

    async def __aenter__(self) -> "LoLEsportsClient":
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={**DEFAULT_HEADERS, "x-api-key": self.api_key},
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    )
    async def _request(self, endpoint: str, params: dict | None = None) -> dict:
        """Make a request to the API with retry logic."""
        async with self.semaphore:
            url = f"{API_BASE_URL}/{endpoint}"
            params = params or {}
            params["hl"] = "en-US"  # Language

            logger.debug(f"Requesting {url} with params {params}")

            response = await self.client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            return data.get("data", data)

    async def get_leagues(self) -> list[League]:
        """Get all available leagues."""
        data = await self._request("getLeagues")
        leagues = []

        for league_data in data.get("leagues", []):
            leagues.append(
                League(
                    id=league_data["id"],
                    slug=league_data["slug"],
                    name=league_data["name"],
                    region=league_data.get("region", ""),
                    image_url=league_data.get("image"),
                    priority=league_data.get("priority", 0),
                )
            )

        return leagues

    async def get_tournaments(self, league_id: str) -> list[Tournament]:
        """Get tournaments for a specific league."""
        data = await self._request(
            "getTournamentsForLeague", params={"leagueId": league_id}
        )
        tournaments = []

        for league in data.get("leagues", []):
            for tournament_data in league.get("tournaments", []):
                start_date = None
                end_date = None

                if tournament_data.get("startDate"):
                    start_date = datetime.fromisoformat(
                        tournament_data["startDate"].replace("Z", "+00:00")
                    )
                if tournament_data.get("endDate"):
                    end_date = datetime.fromisoformat(
                        tournament_data["endDate"].replace("Z", "+00:00")
                    )

                tournaments.append(
                    Tournament(
                        id=tournament_data["id"],
                        slug=tournament_data["slug"],
                        name=tournament_data.get("name", tournament_data["slug"]),
                        league_id=league_id,
                        start_date=start_date,
                        end_date=end_date,
                    )
                )

        return tournaments

    async def get_schedule(
        self,
        league_id: str | None = None,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """
        Get match schedule.

        Returns dict with 'events' list and 'pages' for pagination.
        """
        params = {}
        if league_id:
            params["leagueId"] = league_id
        if page_token:
            params["pageToken"] = page_token

        return await self._request("getSchedule", params=params)

    async def get_event_details(self, match_id: str) -> dict:
        """Get detailed information about a match/event."""
        data = await self._request("getEventDetails", params={"id": match_id})
        return data.get("event", {})

    async def get_games(self, match_id: str) -> list[dict]:
        """Get games (individual maps) for a match."""
        data = await self._request("getGames", params={"id": match_id})
        return data.get("games", [])

    async def get_completed_matches(
        self,
        league_id: str,
        tournament_id: str | None = None,
    ) -> list[Match]:
        """Get all completed matches for a league."""
        matches = []
        page_token = None

        while True:
            schedule = await self.get_schedule(
                league_id=league_id, page_token=page_token
            )

            for event in schedule.get("events", []):
                if event.get("state") != "completed":
                    continue

                if event.get("type") != "match":
                    continue

                match_data = event.get("match", {})
                if not match_data:
                    continue

                if tournament_id and event.get("tournament", {}).get("id") != tournament_id:
                    continue

                teams = match_data.get("teams", [])
                team1 = teams[0] if len(teams) > 0 else {}
                team2 = teams[1] if len(teams) > 1 else {}

                start_time = None
                if event.get("startTime"):
                    start_time = datetime.fromisoformat(
                        event["startTime"].replace("Z", "+00:00")
                    )

                strategy = match_data.get("strategy", {})

                matches.append(
                    Match(
                        id=match_data["id"],
                        state=event["state"],
                        block_name=event.get("blockName"),
                        league_slug=event.get("league", {}).get("slug", ""),
                        tournament_id=event.get("tournament", {}).get("id", ""),
                        strategy_type=strategy.get("type", "bestOf"),
                        strategy_count=strategy.get("count", 1),
                        team1=team1,
                        team2=team2,
                        games=match_data.get("games", []),
                        start_time=start_time,
                    )
                )

            pages = schedule.get("pages", {})
            page_token = pages.get("newer")

            if not page_token:
                break

        return matches


class LoLEsportsFeedClient:
    """Client for live/detailed game stats feed."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "LoLEsportsFeedClient":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def get_window(self, game_id: str, starting_time: str | None = None) -> dict:
        """
        Get a time window of game data.

        Args:
            game_id: The game ID
            starting_time: ISO timestamp to start from (for pagination)

        Returns:
            Dict with frames, game metadata, and participant stats
        """
        url = f"{FEED_BASE_URL}/window/{game_id}"
        params = {}
        if starting_time:
            params["startingTime"] = starting_time

        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def get_details(self, game_id: str, starting_time: str | None = None) -> dict:
        """
        Get detailed game events.

        Args:
            game_id: The game ID
            starting_time: ISO timestamp to start from

        Returns:
            Dict with detailed events (kills, objectives, etc.)
        """
        url = f"{FEED_BASE_URL}/details/{game_id}"
        params = {}
        if starting_time:
            params["startingTime"] = starting_time

        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def get_full_game_data(self, game_id: str) -> dict:
        """
        Get complete game data by paginating through all windows.

        Returns combined data from all time windows.
        """
        all_frames = []
        all_events = []
        game_metadata = None
        participants = None

        starting_time = None

        while True:
            try:
                window = await self.get_window(game_id, starting_time)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    break
                raise

            if game_metadata is None:
                game_metadata = window.get("gameMetadata")
                participants = window.get("participants", [])

            frames = window.get("frames", [])
            if not frames:
                break

            all_frames.extend(frames)

            last_frame = frames[-1]
            starting_time = last_frame.get("rfc460Timestamp")

            if len(frames) < 10:
                break

        starting_time = None
        while True:
            try:
                details = await self.get_details(game_id, starting_time)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    break
                raise

            frames = details.get("frames", [])
            if not frames:
                break

            for frame in frames:
                all_events.extend(frame.get("events", []))

            last_frame = frames[-1]
            starting_time = last_frame.get("rfc460Timestamp")

            if len(frames) < 10:
                break

        return {
            "gameMetadata": game_metadata,
            "participants": participants,
            "frames": all_frames,
            "events": all_events,
        }


async def main():
    """Example usage of the clients."""
    async with LoLEsportsClient() as client:
        leagues = await client.get_leagues()
        print(f"Found {len(leagues)} leagues:")
        for league in leagues[:5]:
            print(f"  - {league.name} ({league.slug})")

        cblol = next((l for l in leagues if l.slug == "cblol"), None)
        if cblol:
            print(f"\nGetting tournaments for {cblol.name}...")
            tournaments = await client.get_tournaments(cblol.id)
            print(f"Found {len(tournaments)} tournaments")

            if tournaments:
                print(f"\nGetting completed matches...")
                matches = await client.get_completed_matches(cblol.id)
                print(f"Found {len(matches)} completed matches")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
