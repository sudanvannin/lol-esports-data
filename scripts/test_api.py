"""
Quick test script to validate LoL Esports API connectivity.
Run: python scripts/test_api.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx

API_BASE_URL = "https://esports-api.lolesports.com/persisted/gw"
API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"


async def test_api():
    """Test API connectivity and basic endpoints."""
    print("=" * 60)
    print("Testing LoL Esports API")
    print("=" * 60)

    headers = {
        "x-api-key": API_KEY,
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        # Test 1: Get Leagues
        print("\n[1] Testing /getLeagues...")
        try:
            response = await client.get(
                f"{API_BASE_URL}/getLeagues",
                params={"hl": "en-US"},
            )
            response.raise_for_status()
            data = response.json()
            leagues = data.get("data", {}).get("leagues", [])
            print(f"    Status: {response.status_code} OK")
            print(f"    Leagues found: {len(leagues)}")

            # Show some leagues
            print("    Sample leagues:")
            for league in leagues[:5]:
                print(f"      - {league['name']} ({league['slug']})")

        except httpx.HTTPStatusError as e:
            print(f"    FAILED: HTTP {e.response.status_code}")
            print(f"    Response: {e.response.text[:200]}")
            return False
        except Exception as e:
            print(f"    FAILED: {e}")
            return False

        # Test 2: Get Schedule
        print("\n[2] Testing /getSchedule...")
        try:
            response = await client.get(
                f"{API_BASE_URL}/getSchedule",
                params={"hl": "en-US"},
            )
            response.raise_for_status()
            data = response.json()
            events = data.get("data", {}).get("schedule", {}).get("events", [])
            print(f"    Status: {response.status_code} OK")
            print(f"    Events found: {len(events)}")

            # Show recent matches
            completed = [e for e in events if e.get("state") == "completed"][:3]
            if completed:
                print("    Recent completed matches:")
                for event in completed:
                    match = event.get("match", {})
                    teams = match.get("teams", [])
                    if len(teams) >= 2:
                        print(f"      - {teams[0].get('name', '?')} vs {teams[1].get('name', '?')}")

        except httpx.HTTPStatusError as e:
            print(f"    FAILED: HTTP {e.response.status_code}")
            return False
        except Exception as e:
            print(f"    FAILED: {e}")
            return False

        # Test 3: Get specific league (CBLOL)
        print("\n[3] Testing tournaments for CBLOL...")
        try:
            cblol = next((l for l in leagues if l["slug"] == "cblol-brazil"), None)
            if cblol:
                response = await client.get(
                    f"{API_BASE_URL}/getTournamentsForLeague",
                    params={"hl": "en-US", "leagueId": cblol["id"]},
                )
                response.raise_for_status()
                data = response.json()
                tournaments = (
                    data.get("data", {})
                    .get("leagues", [{}])[0]
                    .get("tournaments", [])
                )
                print(f"    Status: {response.status_code} OK")
                print(f"    CBLOL tournaments found: {len(tournaments)}")

                if tournaments:
                    print("    Recent tournaments:")
                    for t in tournaments[:3]:
                        print(f"      - {t.get('slug', '?')}")
            else:
                print("    CBLOL not found in leagues")

        except Exception as e:
            print(f"    FAILED: {e}")
            return False

    print("\n" + "=" * 60)
    print("All tests PASSED! API is working correctly.")
    print("=" * 60)
    return True


async def test_feed_api():
    """Test the live stats feed API."""
    print("\n[4] Testing Live Stats Feed API...")

    # This API doesn't require auth but may not have data for old games
    feed_url = "https://feed.lolesports.com/livestats/v1"

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Try to get a known game (this may fail if game is too old)
            response = await client.get(f"{feed_url}/window/1")
            if response.status_code == 404:
                print("    Feed API accessible (404 = no data for test game ID)")
                print("    This is expected - feed data expires after some time")
            else:
                print(f"    Status: {response.status_code}")

        except Exception as e:
            print(f"    Note: {e}")

    return True


if __name__ == "__main__":
    print("\nLoL Esports API Test Suite")
    print("This will test connectivity to the official API\n")

    success = asyncio.run(test_api())
    asyncio.run(test_feed_api())

    sys.exit(0 if success else 1)
