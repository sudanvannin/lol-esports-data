"""Score prematch fair lines for single-game totals markets."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from src.ml.fair_odds import DEFAULT_LATEST_RUN_POINTER
from src.ml.game_totals_fair_lines import DEFAULT_TOTALS_LINES, PrematchGameTotalsScorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Return fair lines for single-game totals markets")
    parser.add_argument("--team1", required=True, help="Team 1 name or key")
    parser.add_argument("--team2", required=True, help="Team 2 name or key")
    parser.add_argument("--match-time", required=True, help="Match time in ISO-8601")
    parser.add_argument("--league-code", required=True, help="League code for the match")
    parser.add_argument("--patch-version", required=True, help="Patch version string")
    parser.add_argument("--kills-line", type=float, default=DEFAULT_TOTALS_LINES["total_kills"])
    parser.add_argument("--dragons-line", type=float, default=DEFAULT_TOTALS_LINES["total_dragons"])
    parser.add_argument("--towers-line", type=float, default=DEFAULT_TOTALS_LINES["total_towers"])
    parser.add_argument("--barons-line", type=float, default=DEFAULT_TOTALS_LINES["total_barons"])
    parser.add_argument("--inhibitors-line", type=float, default=DEFAULT_TOTALS_LINES["total_inhibitors"])
    parser.add_argument(
        "--run-pointer",
        type=Path,
        default=DEFAULT_LATEST_RUN_POINTER,
        help="Path to latest_run.json or another model run pointer",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write the JSON payload")
    args = parser.parse_args()

    scorer = PrematchGameTotalsScorer(run_pointer=args.run_pointer)
    quote = scorer.score_match(
        team1=args.team1,
        team2=args.team2,
        match_time=args.match_time,
        league_code=args.league_code,
        patch_version=args.patch_version,
        market_lines={
            "total_kills": args.kills_line,
            "total_dragons": args.dragons_line,
            "total_towers": args.towers_line,
            "total_barons": args.barons_line,
            "total_inhibitors": args.inhibitors_line,
        },
    )

    json_text = json.dumps(asdict(quote), indent=2, ensure_ascii=False)
    print(json_text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
