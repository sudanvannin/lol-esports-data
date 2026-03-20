"""Tests for temporal baseline utilities."""

from __future__ import annotations

import pandas as pd

from src.ml.prematch_baseline import (
    create_calendar_train_validation_test_split,
    create_temporal_train_holdout_split,
)


def test_temporal_split_keeps_holdout_after_development():
    dates = pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "series_key": [f"s{i}" for i in range(len(dates))],
            "event_date": dates,
            "event_time": dates,
            "label_team1_win": [i % 2 for i in range(len(dates))],
        }
    )

    dev_df, holdout_df, folds = create_temporal_train_holdout_split(
        df,
        holdout_fraction=0.25,
        n_splits=3,
    )

    assert not dev_df.empty
    assert not holdout_df.empty
    assert dev_df["event_date"].max() < holdout_df["event_date"].min()
    assert len(folds) == 3
    for fold in folds:
        assert max(pd.to_datetime(fold.train_dates, utc=True)) < min(
            pd.to_datetime(fold.test_dates, utc=True)
        )


def test_calendar_split_creates_explicit_windows():
    dates = pd.date_range("2023-12-25", periods=450, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "series_key": [f"s{i}" for i in range(len(dates))],
            "event_date": dates,
            "event_time": dates,
            "label_team1_win": [i % 2 for i in range(len(dates))],
        }
    )

    train_df, validation_df, test_df = create_calendar_train_validation_test_split(
        df,
        train_end_date="2023-12-31",
        validation_start_date="2024-01-01",
        validation_end_date="2024-12-31",
        test_start_date="2025-01-01",
    )

    assert not train_df.empty
    assert not validation_df.empty
    assert not test_df.empty
    assert train_df["event_date"].max() < validation_df["event_date"].min()
    assert validation_df["event_date"].max() < test_df["event_date"].min()
