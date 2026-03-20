"""Score a manual upcoming matchup and return model fair odds."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from src.ml.fair_odds import DEFAULT_LATEST_RUN_POINTER, PrematchFairOddsScorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a future match and return fair odds")
    parser.add_argument("--team1", required=True, help="Team 1 name or key")
    parser.add_argument("--team2", required=True, help="Team 2 name or key")
    parser.add_argument("--match-time", required=True, help="Match time in ISO-8601")
    parser.add_argument("--league-code", required=True, help="League code used by the feature table")
    parser.add_argument("--split-name", required=True, help="Split or event name")
    parser.add_argument("--patch-version", required=True, help="Patch version string")
    parser.add_argument("--best-of", type=int, default=5, help="Best-of size for the series")
    parser.add_argument(
        "--playoffs",
        action="store_true",
        help="Mark the series as playoffs/bracket stage",
    )
    parser.add_argument("--team1-odds", type=float, default=None, help="Optional decimal odds for team1")
    parser.add_argument("--team2-odds", type=float, default=None, help="Optional decimal odds for team2")
    parser.add_argument(
        "--run-pointer",
        type=Path,
        default=DEFAULT_LATEST_RUN_POINTER,
        help="Path to latest_run.json or another model run pointer",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write the JSON payload")
    args = parser.parse_args()

    scorer = PrematchFairOddsScorer(run_pointer=args.run_pointer)
    quote = scorer.score_match(
        team1=args.team1,
        team2=args.team2,
        match_time=args.match_time,
        league_code=args.league_code,
        split_name=args.split_name,
        patch_version=args.patch_version,
        best_of=args.best_of,
        playoffs=args.playoffs,
        team1_odds=args.team1_odds,
        team2_odds=args.team2_odds,
    )

    payload = asdict(quote)
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(json_text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
