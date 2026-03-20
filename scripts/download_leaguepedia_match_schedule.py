"""Download historical Leaguepedia MatchSchedule rows for reconciliation."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

API_URL = "https://lol.fandom.com/api.php"
OUTPUT_PATH = Path("data/bronze/leaguepedia/match_results.json")
MAX_RETRIES = 8

MATCH_FIELDS = ",".join(
    [
        "MatchId",
        "Team1",
        "Team2",
        "Winner",
        "Team1Score",
        "Team2Score",
        "DateTime_UTC",
        "BestOf",
        "OverviewPage",
        "Patch",
        "Round",
        "Phase",
        "Tab",
        "MatchDay",
    ]
)


def query_api(
    client: httpx.Client,
    *,
    tables: str,
    fields: str,
    where: str = "",
    order_by: str = "",
    limit: int = 500,
    offset: int = 0,
) -> list[dict]:
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

    last_error: RuntimeError | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        response = client.get(API_URL, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()

        error = payload.get("error")
        if error:
            if error.get("code") == "ratelimited":
                sleep_seconds = min(90, 5 * attempt)
                logger.warning(
                    "Leaguepedia rate limited request offset=%s attempt=%s/%s; sleeping %ss",
                    offset,
                    attempt,
                    MAX_RETRIES,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
                last_error = RuntimeError(error)
                continue
            raise RuntimeError(error)

        rows = []
        for item in payload.get("cargoquery", []):
            rows.append(item.get("title", {}))
        return rows

    raise last_error or RuntimeError("Leaguepedia request failed after retries")


def download_year(client: httpx.Client, year: int) -> list[dict]:
    logger.info("Downloading Leaguepedia MatchSchedule for %s", year)

    all_rows: list[dict] = []
    offset = 0
    limit = 500
    where = (
        f"DateTime_UTC >= '{year}-01-01 00:00:00' "
        f"AND DateTime_UTC < '{year + 1}-01-01 00:00:00'"
    )

    while True:
        rows = query_api(
            client,
            tables="MatchSchedule",
            fields=MATCH_FIELDS,
            where=where,
            order_by="DateTime_UTC ASC",
            limit=limit,
            offset=offset,
        )
        if not rows:
            break

        all_rows.extend(rows)
        offset += limit
        logger.info("  year=%s fetched=%s", year, len(all_rows))

        if len(rows) < limit:
            break

        time.sleep(1.0)

    return all_rows


def load_existing_payload() -> tuple[list[dict], set[int]]:
    """Resume from an existing payload when present."""
    if not OUTPUT_PATH.exists():
        return [], set()

    raw_text = OUTPUT_PATH.read_text(encoding="utf-8").strip()
    if not raw_text:
        return [], set()

    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        return [], set()

    rows = payload.get("rows", [])
    completed_years = {
        int(year)
        for year in payload.get("completed_years", [])
        if str(year).isdigit()
    }
    if not isinstance(rows, list):
        rows = []

    return rows, completed_years


def save_payload(rows: list[dict], completed_years: set[int]) -> None:
    """Persist the accumulated MatchSchedule payload."""
    payload = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": "leaguepedia_matchschedule",
        "row_count": len(rows),
        "completed_years": sorted(completed_years),
        "rows": rows,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical Leaguepedia MatchSchedule rows"
    )
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=datetime.now(UTC).year + 1)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing payload progress and restart the selected year range.",
    )
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.end_year < args.start_year:
        raise ValueError("--end-year must be greater than or equal to --start-year")

    if args.no_resume:
        all_rows: list[dict] = []
        completed_years: set[int] = set()
    else:
        all_rows, completed_years = load_existing_payload()
        if completed_years:
            logger.info(
                "Resuming Leaguepedia download with %s rows across years=%s",
                len(all_rows),
                sorted(completed_years),
            )

    with httpx.Client(
        follow_redirects=True,
        timeout=60,
        headers={"User-Agent": "realtime-gold-validator/1.0"},
    ) as client:
        for year in range(args.start_year, args.end_year + 1):
            if year in completed_years:
                logger.info("Skipping Leaguepedia MatchSchedule for %s (already downloaded)", year)
                continue

            year_rows = download_year(client, year)
            all_rows.extend(year_rows)
            completed_years.add(year)
            save_payload(all_rows, completed_years)
            time.sleep(1.0)
    save_payload(all_rows, completed_years)

    logger.info("Saved %s Leaguepedia rows to %s", len(all_rows), OUTPUT_PATH)


if __name__ == "__main__":
    main()
