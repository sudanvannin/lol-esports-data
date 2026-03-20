"""Manage a simple betting ledger for manual paper trading or small stakes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from src.ml.betting_ledger import (
    DEFAULT_LEDGER_PATH,
    append_bet,
    recent_bets,
    settle_bet,
    summarize_betting_ledger,
)


def _sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the betting ledger")
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=DEFAULT_LEDGER_PATH,
        help="Path to the ledger parquet file",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Append a new bet to the ledger")
    add_parser.add_argument("--match-time", required=True, help="Match time in ISO-8601")
    add_parser.add_argument("--league-code", required=True, help="League code")
    add_parser.add_argument("--team1", required=True, help="Team 1 display name")
    add_parser.add_argument("--team2", required=True, help="Team 2 display name")
    add_parser.add_argument("--market", required=True, help="Market name, e.g. moneyline or total_barons")
    add_parser.add_argument("--selection", required=True, help="Selection, e.g. BLG, over, under")
    add_parser.add_argument("--odds", type=float, required=True, help="Decimal odds taken")
    add_parser.add_argument("--stake", type=float, required=True, help="Stake amount in BRL or chosen unit")
    add_parser.add_argument("--fair-odds", type=float, default=None, help="Model fair odds for the selection")
    add_parser.add_argument("--model-prob", type=float, default=None, help="Model probability for the selection")
    add_parser.add_argument("--line", type=float, default=None, help="Optional market line, e.g. 1.5")
    add_parser.add_argument("--bookmaker", default=None, help="Optional bookmaker name")
    add_parser.add_argument("--model-name", default=None, help="Optional model identifier")
    add_parser.add_argument("--model-run-id", default=None, help="Optional model run id")
    add_parser.add_argument("--snapshot-id", default=None, help="Optional Gold snapshot id")
    add_parser.add_argument("--notes", default=None, help="Optional notes")

    settle_parser = subparsers.add_parser("settle", help="Settle one existing bet")
    settle_parser.add_argument("--bet-id", required=True, help="Bet id returned by the add command")
    settle_parser.add_argument("--result", required=True, choices=["win", "loss", "push", "void"])
    settle_parser.add_argument("--notes", default=None, help="Optional settlement notes")
    settle_parser.add_argument("--settled-at", default=None, help="Optional settlement timestamp in ISO-8601")

    summary_parser = subparsers.add_parser("summary", help="Print ledger summary")
    summary_parser.add_argument("--limit", type=int, default=10, help="Recent bets to show")

    list_parser = subparsers.add_parser("list", help="List recent bets")
    list_parser.add_argument("--limit", type=int, default=20, help="Rows to return")
    list_parser.add_argument("--status", choices=["open", "settled"], default=None, help="Optional status filter")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "add":
        record = append_bet(
            ledger_path=args.ledger_path,
            match_time=args.match_time,
            league_code=args.league_code,
            team1_name=args.team1,
            team2_name=args.team2,
            market=args.market,
            selection=args.selection,
            odds=args.odds,
            stake=args.stake,
            fair_odds=args.fair_odds,
            model_probability=args.model_prob,
            line=args.line,
            bookmaker=args.bookmaker,
            model_name=args.model_name,
            model_run_id=args.model_run_id,
            snapshot_id=args.snapshot_id,
            notes=args.notes,
        )
        payload = {
            "ledger_path": str(args.ledger_path),
            "bet": record,
            "summary": summarize_betting_ledger(args.ledger_path),
        }
        print(json.dumps(_sanitize_for_json(payload), indent=2, ensure_ascii=False))
        return

    if args.command == "settle":
        record = settle_bet(
            bet_id=args.bet_id,
            result=args.result,
            ledger_path=args.ledger_path,
            settled_at=args.settled_at,
            notes=args.notes,
        )
        payload = {
            "ledger_path": str(args.ledger_path),
            "bet": record,
            "summary": summarize_betting_ledger(args.ledger_path),
        }
        print(json.dumps(_sanitize_for_json(payload), indent=2, ensure_ascii=False))
        return

    if args.command == "summary":
        payload = {
            "ledger_path": str(args.ledger_path),
            "summary": summarize_betting_ledger(args.ledger_path),
            "recent_bets": recent_bets(args.ledger_path, limit=args.limit),
        }
        print(json.dumps(_sanitize_for_json(payload), indent=2, ensure_ascii=False))
        return

    if args.command == "list":
        payload = {
            "ledger_path": str(args.ledger_path),
            "bets": recent_bets(args.ledger_path, limit=args.limit, status=args.status),
        }
        print(json.dumps(_sanitize_for_json(payload), indent=2, ensure_ascii=False))
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
