"""Tests for local DuckDB fallback when MotherDuck is unavailable."""

from __future__ import annotations

from pathlib import Path

from web import db


def test_local_parquet_fallback_registers_core_views(monkeypatch):
    monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
    monkeypatch.delenv("MOTHERDUCK_DB", raising=False)
    monkeypatch.chdir(Path.cwd().parent)

    db.reload_data()
    series_df = db.query_df("SELECT COUNT(*) AS total FROM series")
    games_df = db.query_df("SELECT COUNT(*) AS total FROM games")
    players_df = db.query_df("SELECT COUNT(*) AS total FROM players")

    assert int(series_df.iloc[0]["total"]) > 0
    assert int(games_df.iloc[0]["total"]) > 0
    assert int(players_df.iloc[0]["total"]) > 0


def test_upcoming_matches_skip_slow_leaguepedia_fallback_without_token(monkeypatch):
    monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
    monkeypatch.setattr(
        "web.db._load_local_official_schedule_cached",
        lambda: (db._empty_upcoming_matches_df(), db._empty_upcoming_matches_df(), {}),
    )
    monkeypatch.setattr(
        "web.db._load_local_upcoming_matches_cached",
        lambda: (_ for _ in ()).throw(AssertionError("slow fallback should not run")),
    )

    result = db.get_upcoming_matches(limit=8)

    assert result.empty


def test_upcoming_matches_meta_skip_slow_leaguepedia_fallback_without_token(monkeypatch):
    monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
    monkeypatch.setattr(
        "web.db._load_local_official_schedule_cached",
        lambda: (db._empty_upcoming_matches_df(), db._empty_upcoming_matches_df(), {}),
    )
    monkeypatch.setattr(
        "web.db._load_local_upcoming_matches_cached",
        lambda: (_ for _ in ()).throw(AssertionError("slow fallback should not run")),
    )

    meta = db.get_upcoming_matches_meta()

    assert meta["row_count"] == 0
    assert meta["source"] == "local_official_schedule_empty"
