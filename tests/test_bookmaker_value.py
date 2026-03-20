"""Tests for bookmaker value-hunting helpers."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.ml.betting_ledger import load_betting_ledger
from src.ml.bookmaker_value import (
    evaluate_moneyline_sides,
    normalize_moneyline_market_frame,
    recommend_moneyline_side,
    record_recommendations_to_ledger,
)
from src.ml.fair_odds import compare_two_way_market


def _workspace_ledger_path() -> Path:
    base_dir = Path("data/test_runs/bookmaker_value")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{uuid4().hex}.parquet"


def test_normalize_moneyline_market_frame_accepts_common_aliases():
    raw_df = pd.DataFrame(
        [
            {
                "Start Time": "2026-03-16T13:00:00Z",
                "Competition": "First Stand",
                "Home Team": "Bilibili Gaming",
                "Away Team": "BNK FEARX",
                "Home Odds": 1.82,
                "Away Odds": 2.04,
            }
        ]
    )

    normalized = normalize_moneyline_market_frame(raw_df)

    assert normalized.at[0, "match_time"] == "2026-03-16T13:00:00Z"
    assert normalized.at[0, "league"] == "First Stand"
    assert normalized.at[0, "team1"] == "Bilibili Gaming"
    assert normalized.at[0, "team2"] == "BNK FEARX"
    assert round(float(normalized.at[0, "team1_odds"]), 2) == 1.82
    assert round(float(normalized.at[0, "team2_odds"]), 2) == 2.04


def test_recommend_moneyline_side_picks_highest_positive_ev_side():
    comparison = compare_two_way_market(
        team1_model_prob=0.44,
        team2_model_prob=0.56,
        team1_odds=2.10,
        team2_odds=2.20,
    )
    sides = evaluate_moneyline_sides(
        team1_name="Team One",
        team2_name="Team Two",
        team1_model_prob=0.44,
        team2_model_prob=0.56,
        team1_fair_odds=1.0 / 0.44,
        team2_fair_odds=1.0 / 0.56,
        market_comparison=comparison,
    )

    recommendation = recommend_moneyline_side(sides, min_edge=0.01, min_ev=0.03)

    assert recommendation.recommend_bet is True
    assert recommendation.side == "team2"
    assert recommendation.selection == "Team Two"
    assert recommendation.ev_per_unit is not None and recommendation.ev_per_unit > 0.20


def test_record_recommendations_to_ledger_skips_duplicate_open_bets():
    ledger_path = _workspace_ledger_path()
    scored_df = pd.DataFrame(
        [
            {
                "recommend_bet": True,
                "score_available": True,
                "match_time": "2026-03-16T13:00:00Z",
                "league_code": "FST",
                "team1_name": "Bilibili Gaming",
                "team2_name": "BNK FEARX",
                "recommended_selection": "Bilibili Gaming",
                "recommended_odds": 1.91,
                "recommended_fair_odds": 1.71,
                "recommended_model_probability": 1.0 / 1.71,
                "bookmaker": "Bet365",
                "model_name": "logistic_regression",
                "model_run_id": "run_1",
                "snapshot_id": "snapshot_1",
                "notes": "manual import",
                "recommendation_reason": "Highest EV side above thresholds.",
            }
        ]
    )

    first_result = record_recommendations_to_ledger(
        scored_df,
        ledger_path=ledger_path,
        default_stake=1.0,
    )
    second_result = record_recommendations_to_ledger(
        scored_df,
        ledger_path=ledger_path,
        default_stake=1.0,
    )

    ledger_df = load_betting_ledger(ledger_path)

    assert len(first_result["added"]) == 1
    assert len(first_result["skipped"]) == 0
    assert len(second_result["added"]) == 0
    assert len(second_result["skipped"]) == 1
    assert len(ledger_df) == 1
