"""Score imported Bet365 moneylines against the prematch model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from src.ml.bookmaker_value import (
    DEFAULT_BOOKMAKER,
    load_moneyline_market_csv,
    record_recommendations_to_ledger,
    score_moneyline_frame,
)
from src.ml.betting_ledger import DEFAULT_LEDGER_PATH
from src.ml.fair_odds import DEFAULT_LATEST_RUN_POINTER


def _sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _pct_to_decimal(value: float) -> float:
    return float(value) / 100.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare imported Bet365 moneylines against the prematch fair-odds model"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV file with match_time, team1, team2, team1_odds, team2_odds and league or league_code",
    )
    parser.add_argument(
        "--run-pointer",
        type=Path,
        default=DEFAULT_LATEST_RUN_POINTER,
        help="Path to prematch latest_run.json",
    )
    parser.add_argument(
        "--bookmaker",
        default=DEFAULT_BOOKMAKER,
        help="Default bookmaker label when the CSV does not include one",
    )
    parser.add_argument(
        "--min-edge-pct",
        type=float,
        default=2.0,
        help="Minimum edge versus devigged market probability, in percentage points",
    )
    parser.add_argument(
        "--min-ev-pct",
        type=float,
        default=3.0,
        help="Minimum expected value per unit, in percent",
    )
    parser.add_argument(
        "--min-kelly-pct",
        type=float,
        default=0.0,
        help="Minimum full Kelly fraction, in percent of bankroll",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Top recommendations to include in the JSON summary",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save the fully scored CSV",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the JSON summary",
    )
    parser.add_argument(
        "--write-ledger",
        action="store_true",
        help="Append recommended moneyline bets to the betting ledger",
    )
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=DEFAULT_LEDGER_PATH,
        help="Ledger parquet file used when --write-ledger is enabled",
    )
    parser.add_argument(
        "--stake",
        type=float,
        default=1.0,
        help="Default unit stake used for ledger entries when the CSV does not contain a stake column",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_df = load_moneyline_market_csv(args.input)
    scored_df = score_moneyline_frame(
        input_df,
        run_pointer=args.run_pointer,
        min_edge=_pct_to_decimal(args.min_edge_pct),
        min_ev=_pct_to_decimal(args.min_ev_pct),
        min_kelly=_pct_to_decimal(args.min_kelly_pct),
        default_bookmaker=args.bookmaker,
    )

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        scored_df.to_csv(args.output_csv, index=False)

    recommended_df = scored_df.loc[
        scored_df["recommend_bet"].fillna(False) & scored_df["score_available"].fillna(False)
    ].copy()
    top_recommendations = (
        recommended_df.sort_values(
            ["recommended_ev_pct", "recommended_edge_pct"],
            ascending=[False, False],
            kind="stable",
        )
        .head(args.top)
        .to_dict(orient="records")
        if not recommended_df.empty
        else []
    )

    ledger_result = None
    if args.write_ledger:
        ledger_result = record_recommendations_to_ledger(
            scored_df,
            ledger_path=args.ledger_path,
            default_stake=args.stake,
        )

    summary = {
        "input_path": str(args.input),
        "run_pointer": str(args.run_pointer),
        "rows_total": int(len(scored_df)),
        "rows_scored": int(scored_df["score_available"].fillna(False).sum()),
        "rows_failed": int((~scored_df["score_available"].fillna(False)).sum()),
        "recommendations": int(recommended_df.shape[0]),
        "bookmaker_default": args.bookmaker,
        "thresholds": {
            "min_edge_pct": float(args.min_edge_pct),
            "min_ev_pct": float(args.min_ev_pct),
            "min_kelly_pct": float(args.min_kelly_pct),
        },
        "top_recommendations": top_recommendations,
        "output_csv": str(args.output_csv) if args.output_csv is not None else None,
        "ledger": ledger_result,
    }

    json_text = json.dumps(_sanitize_for_json(summary), indent=2, ensure_ascii=False)
    print(json_text)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
