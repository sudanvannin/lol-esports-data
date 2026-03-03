"""List all available leagues."""
import asyncio
import httpx

async def get_all_leagues():
    headers = {"x-api-key": "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"}
    async with httpx.AsyncClient(headers=headers) as client:
        r = await client.get(
            "https://esports-api.lolesports.com/persisted/gw/getLeagues",
            params={"hl": "en-US"}
        )
        leagues = r.json()["data"]["leagues"]
        print("Ligas disponiveis:")
        print("-" * 70)
        for l in leagues:
            slug = l["slug"]
            name = l["name"]
            region = l.get("region", "")
            print(f"  {slug:30} | {name:25} | {region}")

asyncio.run(get_all_leagues())
