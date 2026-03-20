"""Tests for local DuckDB fallback when MotherDuck is unavailable."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from web import db


def _build_local_silver_fixture(root: Path) -> Path:
    silver_dir = root / "silver"
    games_dir = silver_dir / "games" / "league=TEST"
    players_dir = silver_dir / "players" / "league=TEST"
    games_dir.mkdir(parents=True, exist_ok=True)
    players_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"game_id": "g1"}]).to_parquet(
        games_dir / "games.parquet", index=False
    )
    pd.DataFrame([{"player_id": "p1"}]).to_parquet(
        players_dir / "players.parquet", index=False
    )
    pd.DataFrame([{"series_id": "s1"}]).to_parquet(
        silver_dir / "series.parquet", index=False
    )
    pd.DataFrame([{"champion_name": "Ahri"}]).to_parquet(
        silver_dir / "champions.parquet", index=False
    )

    return silver_dir


def test_local_parquet_fallback_registers_core_views(monkeypatch, tmp_path):
    monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
    monkeypatch.delenv("MOTHERDUCK_DB", raising=False)
    monkeypatch.setattr(db, "LOCAL_SILVER_DIR", _build_local_silver_fixture(tmp_path))

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


def test_upcoming_matches_meta_skip_slow_leaguepedia_fallback_without_token(
    monkeypatch,
):
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
