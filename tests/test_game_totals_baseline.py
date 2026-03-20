"""Tests for game totals baseline helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.game_totals_baseline import (
    _estimate_overdispersion_alpha,
    build_game_totals_team_history_frame,
    build_upcoming_game_totals_feature_row,
)


def test_estimate_overdispersion_alpha_is_zero_when_residuals_are_small():
    y_true = pd.Series([10.0, 12.0, 11.0, 9.0])
    predictions = np.array([10.5, 11.5, 10.8, 9.5])

    alpha = _estimate_overdispersion_alpha(y_true, predictions)

    assert alpha >= 0.0


def test_estimate_overdispersion_alpha_grows_with_large_residuals():
    y_true = pd.Series([10.0, 30.0, 5.0, 40.0])
    predictions = np.array([15.0, 15.0, 15.0, 15.0])

    alpha = _estimate_overdispersion_alpha(y_true, predictions)

    assert alpha > 0.0


def test_build_upcoming_game_totals_feature_row_uses_prior_games_only():
    games_df = pd.DataFrame(
        [
            {
                "snapshot_id": "snap",
                "game_id": "g1",
                "event_time": pd.Timestamp("2024-01-01T12:00:00Z"),
                "event_date": pd.Timestamp("2024-01-01T00:00:00Z"),
                "league_code": "CBLOL",
                "split_name": "CBLOL",
                "patch_version": "14.1",
                "playoffs": False,
                "team1_key": "aaa",
                "team1_name": "AAA",
                "team2_key": "bbb",
                "team2_name": "BBB",
                "game_length_seconds": 1800.0,
                "target_total_kills": 30.0,
                "target_total_dragons": 5.0,
                "target_total_towers": 11.0,
                "target_total_barons": 1.0,
                "target_total_inhibitors": 2.0,
                "team1_team_kills": 18.0,
                "team2_team_kills": 12.0,
                "team1_dragons": 3.0,
                "team2_dragons": 2.0,
                "team1_towers": 7.0,
                "team2_towers": 4.0,
                "team1_barons": 1.0,
                "team2_barons": 0.0,
                "team1_inhibitors": 2.0,
                "team2_inhibitors": 0.0,
            },
            {
                "snapshot_id": "snap",
                "game_id": "g2",
                "event_time": pd.Timestamp("2024-01-03T12:00:00Z"),
                "event_date": pd.Timestamp("2024-01-03T00:00:00Z"),
                "league_code": "CBLOL",
                "split_name": "CBLOL",
                "patch_version": "14.1",
                "playoffs": False,
                "team1_key": "aaa",
                "team1_name": "AAA",
                "team2_key": "ccc",
                "team2_name": "CCC",
                "game_length_seconds": 1950.0,
                "target_total_kills": 24.0,
                "target_total_dragons": 4.0,
                "target_total_towers": 10.0,
                "target_total_barons": 1.0,
                "target_total_inhibitors": 1.0,
                "team1_team_kills": 14.0,
                "team2_team_kills": 10.0,
                "team1_dragons": 2.0,
                "team2_dragons": 2.0,
                "team1_towers": 6.0,
                "team2_towers": 4.0,
                "team1_barons": 1.0,
                "team2_barons": 0.0,
                "team1_inhibitors": 1.0,
                "team2_inhibitors": 0.0,
            },
        ]
    )
    team_history_df = build_game_totals_team_history_frame(games_df)

    feature_row, metadata = build_upcoming_game_totals_feature_row(
        games_df,
        team_history_df,
        snapshot_id="snap",
        team1_key="aaa",
        team1_name="AAA",
        team2_key="bbb",
        team2_name="BBB",
        event_time=pd.Timestamp("2024-01-05T12:00:00Z"),
        league_code="CBLOL",
        split_name="CBLOL",
        patch_version="14.1",
    )

    assert len(feature_row) == 1
    assert metadata["team1_prior_game_count"] == 2
    assert metadata["team2_prior_game_count"] == 1
    assert metadata["baseline_sample_count"] == 2
    assert feature_row.iloc[0]["team1_recent5_total_kills_mean"] == 27.0
