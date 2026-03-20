"""Tests for the betting ledger helpers."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.ml.betting_ledger import append_bet, load_betting_ledger, settle_bet, summarize_betting_ledger


def _workspace_ledger_path() -> Path:
    base_dir = Path("data/test_runs/betting_ledger")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{uuid4().hex}.parquet"


def test_append_bet_creates_ledger_and_derives_ev():
    ledger_path = _workspace_ledger_path()

    record = append_bet(
        ledger_path=ledger_path,
        match_time="2026-03-16T13:00:00Z",
        league_code="FST",
        team1_name="Bilibili Gaming",
        team2_name="BNK FEARX",
        market="total_barons",
        selection="over",
        line=1.5,
        odds=2.20,
        fair_odds=1.84,
        stake=10.0,
        bookmaker="Example",
    )

    ledger_df = load_betting_ledger(ledger_path)
    assert len(ledger_df) == 1
    assert record["status"] == "open"
    assert record["result"] == "open"
    assert round(float(record["model_probability"]), 4) == round(1.0 / 1.84, 4)
    assert round(float(record["ev_per_unit"]), 4) == round((1.0 / 1.84) * 2.20 - 1.0, 4)


def test_settle_bet_updates_profit_loss_and_summary():
    ledger_path = _workspace_ledger_path()
    record = append_bet(
        ledger_path=ledger_path,
        match_time="2026-03-16T13:00:00Z",
        league_code="FST",
        team1_name="Bilibili Gaming",
        team2_name="BNK FEARX",
        market="moneyline",
        selection="Bilibili Gaming",
        odds=1.42,
        fair_odds=1.25,
        stake=10.0,
    )

    settled = settle_bet(record["bet_id"], "win", ledger_path=ledger_path)
    summary = summarize_betting_ledger(ledger_path)

    assert settled["status"] == "settled"
    assert settled["result"] == "win"
    assert round(float(settled["profit_loss"]), 2) == 4.20
    assert round(float(settled["roi_pct"]), 2) == 42.00
    assert summary["wins"] == 1
    assert summary["losses"] == 0
    assert round(float(summary["realized_profit_loss"]), 2) == 4.20
    assert round(float(summary["realized_roi_pct"]), 2) == 42.00
