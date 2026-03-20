"""Simple betting ledger for manual paper trading and small real-money tracking."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .fair_odds import _probability_to_fair_odds

DEFAULT_LEDGER_PATH = Path("data/tracking/betting_ledger.parquet")
OPEN_RESULT = "open"
SETTLED_RESULTS = {"win", "loss", "push", "void"}
LEDGER_COLUMNS = [
    "bet_id",
    "created_at",
    "settled_at",
    "status",
    "result",
    "match_time",
    "league_code",
    "team1_name",
    "team2_name",
    "market",
    "selection",
    "line",
    "bookmaker",
    "odds",
    "fair_odds",
    "model_probability",
    "market_implied_probability",
    "edge_probability",
    "ev_per_unit",
    "stake",
    "potential_payout",
    "potential_profit",
    "payout",
    "profit_loss",
    "roi_pct",
    "model_name",
    "model_run_id",
    "snapshot_id",
    "notes",
]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_bet_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"bet_{timestamp}_{uuid.uuid4().hex[:8]}"


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def load_betting_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> pd.DataFrame:
    """Load the betting ledger if it exists, otherwise return an empty frame."""
    if not ledger_path.exists():
        return _empty_ledger()

    ledger_df = pd.read_parquet(ledger_path)
    missing_columns = [column for column in LEDGER_COLUMNS if column not in ledger_df.columns]
    for column in missing_columns:
        ledger_df[column] = np.nan
    return ledger_df[LEDGER_COLUMNS].copy()


def save_betting_ledger(ledger_df: pd.DataFrame, ledger_path: Path = DEFAULT_LEDGER_PATH) -> Path:
    """Persist the ledger to parquet."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_df[LEDGER_COLUMNS].to_parquet(ledger_path, index=False)
    return ledger_path


def create_bet_record(
    *,
    match_time: str,
    league_code: str,
    team1_name: str,
    team2_name: str,
    market: str,
    selection: str,
    odds: float,
    stake: float,
    fair_odds: float | None = None,
    model_probability: float | None = None,
    line: float | None = None,
    bookmaker: str | None = None,
    model_name: str | None = None,
    model_run_id: str | None = None,
    snapshot_id: str | None = None,
    notes: str | None = None,
    created_at: str | None = None,
    bet_id: str | None = None,
) -> dict[str, Any]:
    """Create one normalized ledger record."""
    if odds <= 1.0:
        raise ValueError("odds must be greater than 1.0")
    if stake <= 0.0:
        raise ValueError("stake must be greater than 0.0")
    if fair_odds is None and model_probability is None:
        raise ValueError("fair_odds or model_probability must be provided")
    if fair_odds is not None and fair_odds <= 1.0:
        raise ValueError("fair_odds must be greater than 1.0")
    if model_probability is not None and not 0.0 < model_probability < 1.0:
        raise ValueError("model_probability must be between 0 and 1")

    normalized_model_probability = (
        float(model_probability) if model_probability is not None else float(1.0 / float(fair_odds))
    )
    normalized_fair_odds = (
        float(fair_odds) if fair_odds is not None else float(_probability_to_fair_odds(normalized_model_probability))
    )
    market_implied_probability = float(1.0 / float(odds))
    edge_probability = float(normalized_model_probability - market_implied_probability)
    ev_per_unit = float(normalized_model_probability * float(odds) - 1.0)
    created_at_value = created_at or _utc_now_iso()

    return {
        "bet_id": bet_id or _new_bet_id(),
        "created_at": created_at_value,
        "settled_at": None,
        "status": OPEN_RESULT,
        "result": OPEN_RESULT,
        "match_time": str(match_time),
        "league_code": str(league_code),
        "team1_name": str(team1_name),
        "team2_name": str(team2_name),
        "market": str(market),
        "selection": str(selection),
        "line": float(line) if line is not None else np.nan,
        "bookmaker": str(bookmaker) if bookmaker else "",
        "odds": float(odds),
        "fair_odds": normalized_fair_odds,
        "model_probability": normalized_model_probability,
        "market_implied_probability": market_implied_probability,
        "edge_probability": edge_probability,
        "ev_per_unit": ev_per_unit,
        "stake": float(stake),
        "potential_payout": float(stake * odds),
        "potential_profit": float(stake * (odds - 1.0)),
        "payout": np.nan,
        "profit_loss": np.nan,
        "roi_pct": np.nan,
        "model_name": str(model_name) if model_name else "",
        "model_run_id": str(model_run_id) if model_run_id else "",
        "snapshot_id": str(snapshot_id) if snapshot_id else "",
        "notes": str(notes) if notes else "",
    }


def append_bet(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    **bet_kwargs: Any,
) -> dict[str, Any]:
    """Append a new bet to the ledger and persist it."""
    ledger_df = load_betting_ledger(ledger_path)
    record = create_bet_record(**bet_kwargs)
    if ledger_df.empty:
        updated_df = pd.DataFrame([record], columns=LEDGER_COLUMNS)
    else:
        updated_df = pd.concat([ledger_df, pd.DataFrame([record])], ignore_index=True)
    save_betting_ledger(updated_df, ledger_path)
    return record


def settle_bet(
    bet_id: str,
    result: str,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    settled_at: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Settle one existing bet and persist the updated ledger."""
    normalized_result = str(result).lower().strip()
    if normalized_result not in SETTLED_RESULTS:
        raise ValueError(f"result must be one of {sorted(SETTLED_RESULTS)}")

    ledger_df = load_betting_ledger(ledger_path)
    if ledger_df.empty:
        raise ValueError("ledger is empty")

    matching_index = ledger_df.index[ledger_df["bet_id"] == bet_id]
    if len(matching_index) == 0:
        raise ValueError(f"bet_id not found: {bet_id}")
    row_index = matching_index[0]
    if str(ledger_df.at[row_index, "status"]) != OPEN_RESULT:
        raise ValueError(f"bet already settled: {bet_id}")

    stake = float(ledger_df.at[row_index, "stake"])
    odds = float(ledger_df.at[row_index, "odds"])
    if normalized_result == "win":
        payout = float(stake * odds)
        profit_loss = float(stake * (odds - 1.0))
    elif normalized_result == "loss":
        payout = 0.0
        profit_loss = float(-stake)
    else:
        payout = float(stake)
        profit_loss = 0.0

    ledger_df.at[row_index, "settled_at"] = settled_at or _utc_now_iso()
    ledger_df.at[row_index, "status"] = "settled"
    ledger_df.at[row_index, "result"] = normalized_result
    ledger_df.at[row_index, "payout"] = payout
    ledger_df.at[row_index, "profit_loss"] = profit_loss
    ledger_df.at[row_index, "roi_pct"] = float((profit_loss / stake) * 100.0)
    if notes:
        existing_notes = str(ledger_df.at[row_index, "notes"] or "").strip()
        ledger_df.at[row_index, "notes"] = notes if not existing_notes else f"{existing_notes} | {notes}"

    save_betting_ledger(ledger_df, ledger_path)
    return ledger_df.loc[row_index, LEDGER_COLUMNS].to_dict()


def summarize_betting_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> dict[str, Any]:
    """Return a compact summary for the whole betting ledger."""
    ledger_df = load_betting_ledger(ledger_path)
    if ledger_df.empty:
        return {
            "total_bets": 0,
            "open_bets": 0,
            "settled_bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "voids": 0,
            "open_stake": 0.0,
            "total_staked_settled": 0.0,
            "realized_profit_loss": 0.0,
            "realized_roi_pct": 0.0,
            "hit_rate_pct": 0.0,
            "average_edge_probability": 0.0,
            "expected_value_open": 0.0,
        }

    open_df = ledger_df.loc[ledger_df["status"] == OPEN_RESULT].copy()
    settled_df = ledger_df.loc[ledger_df["status"] == "settled"].copy()

    wins = int((settled_df["result"] == "win").sum())
    losses = int((settled_df["result"] == "loss").sum())
    pushes = int((settled_df["result"] == "push").sum())
    voids = int((settled_df["result"] == "void").sum())
    total_staked_settled = float(settled_df["stake"].fillna(0.0).sum())
    realized_profit_loss = float(settled_df["profit_loss"].fillna(0.0).sum())
    realized_roi_pct = float((realized_profit_loss / total_staked_settled) * 100.0) if total_staked_settled > 0.0 else 0.0
    hit_denominator = wins + losses
    hit_rate_pct = float((wins / hit_denominator) * 100.0) if hit_denominator > 0 else 0.0
    average_edge_probability = float(ledger_df["edge_probability"].dropna().mean()) if ledger_df["edge_probability"].notna().any() else 0.0
    expected_value_open = float((open_df["ev_per_unit"].fillna(0.0) * open_df["stake"].fillna(0.0)).sum())

    return {
        "total_bets": int(len(ledger_df)),
        "open_bets": int(len(open_df)),
        "settled_bets": int(len(settled_df)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "voids": voids,
        "open_stake": float(open_df["stake"].fillna(0.0).sum()),
        "total_staked_settled": total_staked_settled,
        "realized_profit_loss": realized_profit_loss,
        "realized_roi_pct": realized_roi_pct,
        "hit_rate_pct": hit_rate_pct,
        "average_edge_probability": average_edge_probability,
        "expected_value_open": expected_value_open,
    }


def recent_bets(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    *,
    limit: int = 20,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Return recent bets for quick inspection."""
    ledger_df = load_betting_ledger(ledger_path)
    if status is not None:
        ledger_df = ledger_df.loc[ledger_df["status"] == status].copy()
    if ledger_df.empty:
        return []
    return (
        ledger_df.sort_values(["created_at", "bet_id"], ascending=[False, False], kind="stable")
        .head(limit)
        .replace({np.nan: None})
        .to_dict(orient="records")
    )


def summary_json(ledger_path: Path = DEFAULT_LEDGER_PATH) -> str:
    """Render a JSON summary for CLI usage."""
    payload = {
        "ledger_path": str(ledger_path),
        "summary": summarize_betting_ledger(ledger_path),
        "recent_bets": recent_bets(ledger_path, limit=10),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
