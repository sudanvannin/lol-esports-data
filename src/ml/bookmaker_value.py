"""Compare imported bookmaker moneylines against the prematch fair-odds model."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.betting_ledger import (
    DEFAULT_LEDGER_PATH,
    OPEN_RESULT,
    create_bet_record,
    load_betting_ledger,
    save_betting_ledger,
)
from src.ml.fair_odds import (
    DEFAULT_LATEST_RUN_POINTER,
    MarketOddsComparison,
    PrematchFairOddsScorer,
)
from web.prob_win import build_match_context

DEFAULT_BOOKMAKER = "Bet365"
DEFAULT_MARKET = "moneyline"
MONEYLINE_REQUIRED_COLUMNS = ("match_time", "team1", "team2", "team1_odds", "team2_odds")
MONEYLINE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "match_id": ("match_id", "id"),
    "match_time": ("match_time", "start_time", "kickoff_time", "commence_time", "date_time"),
    "league": ("league", "competition", "tournament_league"),
    "event_name": ("event_name", "event", "tournament", "competition_name"),
    "phase_label": ("phase_label", "phase", "stage", "round"),
    "team1": ("team1", "home_team", "team_a", "participant1", "selection1"),
    "team2": ("team2", "away_team", "team_b", "participant2", "selection2"),
    "team1_odds": ("team1_odds", "odds_team1", "home_odds", "team_a_odds", "odds1"),
    "team2_odds": ("team2_odds", "odds_team2", "away_odds", "team_b_odds", "odds2"),
    "best_of": ("best_of", "bo", "series_best_of"),
    "patch": ("patch", "patch_version"),
    "league_code": ("league_code",),
    "split_name": ("split_name", "split"),
    "playoffs": ("playoffs", "is_playoffs"),
    "bookmaker": ("bookmaker", "sportsbook", "book"),
    "stake": ("stake", "unit_stake", "bet_size"),
    "notes": ("notes", "comment", "memo"),
}


@dataclass(slots=True)
class MoneylineSideEvaluation:
    """Single-side value summary for a two-way moneyline."""

    side: str
    selection: str
    odds: float
    model_probability: float
    fair_odds: float
    implied_probability_raw: float
    implied_probability_devig: float
    edge_vs_devig: float
    ev_per_unit: float
    kelly_fraction: float


@dataclass(slots=True)
class MoneylineRecommendation:
    """Recommended side for a two-way moneyline, if any."""

    recommend_bet: bool
    reason: str
    side: str | None = None
    selection: str | None = None
    odds: float | None = None
    model_probability: float | None = None
    fair_odds: float | None = None
    implied_probability_devig: float | None = None
    edge_vs_devig: float | None = None
    ev_per_unit: float | None = None
    kelly_fraction: float | None = None


def _normalize_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _string_or_empty(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse playoffs flag: {value!r}")


def _kelly_fraction(probability: float, odds: float) -> float:
    if odds <= 1.0:
        raise ValueError("odds must be greater than 1.0")
    b = float(odds) - 1.0
    raw_fraction = ((float(probability) * float(odds)) - 1.0) / b
    return max(0.0, float(raw_fraction))


def normalize_moneyline_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename common CSV headers into the canonical moneyline schema."""
    normalized_to_original = {
        _normalize_column_name(column): str(column) for column in frame.columns
    }
    rename_map: dict[str, str] = {}
    for canonical_name, aliases in MONEYLINE_COLUMN_ALIASES.items():
        for alias in aliases:
            original_name = normalized_to_original.get(alias)
            if original_name is not None:
                rename_map[original_name] = canonical_name
                break

    normalized = frame.rename(columns=rename_map).copy()

    missing_required = [
        column for column in MONEYLINE_REQUIRED_COLUMNS if column not in normalized.columns
    ]
    if missing_required:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_required)
            + ". Supported aliases: "
            + ", ".join(sorted(frame.columns))
        )

    if "league" not in normalized.columns and "league_code" not in normalized.columns:
        raise ValueError("Missing required context column: league or league_code")

    for numeric_column in ("team1_odds", "team2_odds", "best_of", "stake"):
        if numeric_column in normalized.columns:
            normalized[numeric_column] = pd.to_numeric(
                normalized[numeric_column], errors="coerce"
            )

    if "playoffs" in normalized.columns:
        normalized["playoffs"] = normalized["playoffs"].map(_coerce_optional_bool)

    return normalized.reset_index(drop=True)


def evaluate_moneyline_sides(
    *,
    team1_name: str,
    team2_name: str,
    team1_model_prob: float,
    team2_model_prob: float,
    team1_fair_odds: float,
    team2_fair_odds: float,
    market_comparison: MarketOddsComparison,
) -> list[MoneylineSideEvaluation]:
    """Return value metrics for both sides of a two-way moneyline."""
    return [
        MoneylineSideEvaluation(
            side="team1",
            selection=team1_name,
            odds=float(market_comparison.team1_odds),
            model_probability=float(team1_model_prob),
            fair_odds=float(team1_fair_odds),
            implied_probability_raw=float(market_comparison.team1_implied_prob_raw),
            implied_probability_devig=float(market_comparison.team1_implied_prob_devig),
            edge_vs_devig=float(market_comparison.team1_edge_vs_devig),
            ev_per_unit=float(market_comparison.team1_ev_per_unit),
            kelly_fraction=_kelly_fraction(team1_model_prob, market_comparison.team1_odds),
        ),
        MoneylineSideEvaluation(
            side="team2",
            selection=team2_name,
            odds=float(market_comparison.team2_odds),
            model_probability=float(team2_model_prob),
            fair_odds=float(team2_fair_odds),
            implied_probability_raw=float(market_comparison.team2_implied_prob_raw),
            implied_probability_devig=float(market_comparison.team2_implied_prob_devig),
            edge_vs_devig=float(market_comparison.team2_edge_vs_devig),
            ev_per_unit=float(market_comparison.team2_ev_per_unit),
            kelly_fraction=_kelly_fraction(team2_model_prob, market_comparison.team2_odds),
        ),
    ]


def recommend_moneyline_side(
    side_evaluations: list[MoneylineSideEvaluation],
    *,
    min_edge: float = 0.0,
    min_ev: float = 0.0,
    min_kelly: float = 0.0,
) -> MoneylineRecommendation:
    """Pick the strongest moneyline side that clears the configured thresholds."""
    eligible = [
        evaluation
        for evaluation in side_evaluations
        if evaluation.edge_vs_devig >= min_edge
        and evaluation.ev_per_unit >= min_ev
        and evaluation.kelly_fraction >= min_kelly
    ]
    if not eligible:
        return MoneylineRecommendation(
            recommend_bet=False,
            reason="No side cleared the edge/EV/Kelly thresholds.",
        )

    best_side = max(
        eligible,
        key=lambda evaluation: (
            evaluation.ev_per_unit,
            evaluation.edge_vs_devig,
            evaluation.kelly_fraction,
            evaluation.model_probability,
        ),
    )
    return MoneylineRecommendation(
        recommend_bet=True,
        reason="Highest EV side above thresholds.",
        side=best_side.side,
        selection=best_side.selection,
        odds=best_side.odds,
        model_probability=best_side.model_probability,
        fair_odds=best_side.fair_odds,
        implied_probability_devig=best_side.implied_probability_devig,
        edge_vs_devig=best_side.edge_vs_devig,
        ev_per_unit=best_side.ev_per_unit,
        kelly_fraction=best_side.kelly_fraction,
    )


def _resolve_market_context(row: dict[str, Any]) -> dict[str, Any]:
    raw_match = {
        "match_id": _string_or_empty(row.get("match_id")),
        "match_time": row["match_time"],
        "league": _string_or_empty(row.get("league"))
        or _string_or_empty(row.get("league_code")),
        "event_name": _string_or_empty(row.get("event_name"))
        or _string_or_empty(row.get("league"))
        or _string_or_empty(row.get("league_code")),
        "phase_label": _string_or_empty(row.get("phase_label")),
        "team1": _string_or_empty(row.get("team1")),
        "team2": _string_or_empty(row.get("team2")),
        "best_of": row.get("best_of"),
        "patch": _string_or_empty(row.get("patch")),
    }
    context = build_match_context(raw_match)

    explicit_league_code = _string_or_empty(row.get("league_code"))
    explicit_split_name = _string_or_empty(row.get("split_name"))
    explicit_patch_version = _string_or_empty(row.get("patch"))
    explicit_playoffs = row.get("playoffs")
    explicit_best_of = row.get("best_of")

    if explicit_league_code:
        context["league_code"] = explicit_league_code
    if explicit_split_name:
        context["split_name"] = explicit_split_name
    if explicit_patch_version:
        context["patch_version"] = explicit_patch_version
    if explicit_playoffs is not None and not pd.isna(explicit_playoffs):
        context["playoffs"] = bool(explicit_playoffs)
    if explicit_best_of is not None and not pd.isna(explicit_best_of):
        context["best_of"] = max(int(explicit_best_of), 1)

    return context


def score_moneyline_row(
    row: dict[str, Any],
    *,
    scorer: PrematchFairOddsScorer,
    min_edge: float = 0.0,
    min_ev: float = 0.0,
    min_kelly: float = 0.0,
    default_bookmaker: str = DEFAULT_BOOKMAKER,
) -> dict[str, Any]:
    """Score one imported moneyline row against the prematch model."""
    bookmaker = _string_or_empty(row.get("bookmaker")) or default_bookmaker
    base_payload = {
        "match_id": _string_or_empty(row.get("match_id")),
        "match_time": _string_or_empty(row.get("match_time")),
        "league": _string_or_empty(row.get("league")),
        "event_name": _string_or_empty(row.get("event_name")),
        "phase_label": _string_or_empty(row.get("phase_label")),
        "input_team1": _string_or_empty(row.get("team1")),
        "input_team2": _string_or_empty(row.get("team2")),
        "team1_odds": float(row.get("team1_odds")),
        "team2_odds": float(row.get("team2_odds")),
        "bookmaker": bookmaker,
        "notes": _string_or_empty(row.get("notes")),
        "stake": (
            float(row["stake"])
            if row.get("stake") is not None and not pd.isna(row.get("stake"))
            else None
        ),
    }

    try:
        context = _resolve_market_context(row)
        quote = scorer.score_match(
            team1=str(row["team1"]),
            team2=str(row["team2"]),
            match_time=context["match_time_iso"],
            league_code=str(context["league_code"]),
            split_name=str(context["split_name"]),
            patch_version=str(context["patch_version"]),
            best_of=int(context["best_of"]),
            playoffs=bool(context["playoffs"]),
            team1_odds=float(row["team1_odds"]),
            team2_odds=float(row["team2_odds"]),
        )
        if quote.market_comparison is None:
            raise ValueError("market_comparison was not generated")

        side_evaluations = evaluate_moneyline_sides(
            team1_name=quote.team1_name,
            team2_name=quote.team2_name,
            team1_model_prob=quote.team1_win_prob,
            team2_model_prob=quote.team2_win_prob,
            team1_fair_odds=quote.team1_fair_odds,
            team2_fair_odds=quote.team2_fair_odds,
            market_comparison=quote.market_comparison,
        )
        recommendation = recommend_moneyline_side(
            side_evaluations,
            min_edge=min_edge,
            min_ev=min_ev,
            min_kelly=min_kelly,
        )
    except Exception as exc:
        return {
            **base_payload,
            "score_available": False,
            "score_error": str(exc),
            "recommend_bet": False,
            "recommendation_reason": "Scoring failed.",
        }

    team1_side, team2_side = side_evaluations
    return {
        **base_payload,
        "score_available": True,
        "score_error": "",
        "league_code": str(context["league_code"]),
        "split_name": str(context["split_name"]),
        "patch_version": str(context["patch_version"]),
        "best_of": int(context["best_of"]),
        "playoffs": bool(context["playoffs"]),
        "team1_name": quote.team1_name,
        "team2_name": quote.team2_name,
        "team1_key": quote.team1_key,
        "team2_key": quote.team2_key,
        "team1_model_prob": float(quote.team1_win_prob),
        "team2_model_prob": float(quote.team2_win_prob),
        "team1_fair_odds": float(quote.team1_fair_odds),
        "team2_fair_odds": float(quote.team2_fair_odds),
        "overround_pct": float(quote.market_comparison.overround * 100.0),
        "team1_implied_prob_raw": float(team1_side.implied_probability_raw),
        "team2_implied_prob_raw": float(team2_side.implied_probability_raw),
        "team1_implied_prob_devig": float(team1_side.implied_probability_devig),
        "team2_implied_prob_devig": float(team2_side.implied_probability_devig),
        "team1_edge_pct": float(team1_side.edge_vs_devig * 100.0),
        "team2_edge_pct": float(team2_side.edge_vs_devig * 100.0),
        "team1_ev_pct": float(team1_side.ev_per_unit * 100.0),
        "team2_ev_pct": float(team2_side.ev_per_unit * 100.0),
        "team1_kelly_pct": float(team1_side.kelly_fraction * 100.0),
        "team2_kelly_pct": float(team2_side.kelly_fraction * 100.0),
        "recommend_bet": bool(recommendation.recommend_bet),
        "recommendation_reason": recommendation.reason,
        "recommended_side": recommendation.side or "",
        "recommended_selection": recommendation.selection or "",
        "recommended_odds": recommendation.odds,
        "recommended_model_probability": recommendation.model_probability,
        "recommended_fair_odds": recommendation.fair_odds,
        "recommended_implied_probability_devig": recommendation.implied_probability_devig,
        "recommended_edge_pct": (
            float(recommendation.edge_vs_devig * 100.0)
            if recommendation.edge_vs_devig is not None
            else None
        ),
        "recommended_ev_pct": (
            float(recommendation.ev_per_unit * 100.0)
            if recommendation.ev_per_unit is not None
            else None
        ),
        "recommended_kelly_pct": (
            float(recommendation.kelly_fraction * 100.0)
            if recommendation.kelly_fraction is not None
            else None
        ),
        "model_name": quote.model_name,
        "model_run_id": quote.run_id,
        "snapshot_id": quote.snapshot_id,
        "warnings_json": json.dumps(quote.warnings, ensure_ascii=False),
        "warnings_count": len(quote.warnings),
    }


def score_moneyline_frame(
    frame: pd.DataFrame,
    *,
    run_pointer: Path = DEFAULT_LATEST_RUN_POINTER,
    min_edge: float = 0.0,
    min_ev: float = 0.0,
    min_kelly: float = 0.0,
    default_bookmaker: str = DEFAULT_BOOKMAKER,
) -> pd.DataFrame:
    """Score every row in an imported bookmaker moneyline sheet."""
    normalized = normalize_moneyline_market_frame(frame)
    scorer = PrematchFairOddsScorer(run_pointer=run_pointer)
    records = [
        score_moneyline_row(
            row,
            scorer=scorer,
            min_edge=min_edge,
            min_ev=min_ev,
            min_kelly=min_kelly,
            default_bookmaker=default_bookmaker,
        )
        for row in normalized.to_dict(orient="records")
    ]
    scored_df = pd.DataFrame(records)
    if scored_df.empty:
        return scored_df

    sort_columns = [
        column
        for column in ("recommend_bet", "recommended_ev_pct", "recommended_edge_pct")
        if column in scored_df.columns
    ]
    if sort_columns:
        scored_df = scored_df.sort_values(
            sort_columns,
            ascending=[False, False, False][: len(sort_columns)],
            kind="stable",
        ).reset_index(drop=True)
    return scored_df


def load_moneyline_market_csv(csv_path: Path) -> pd.DataFrame:
    """Load one bookmaker import file."""
    return pd.read_csv(csv_path)


def _ledger_duplicate_mask(ledger_df: pd.DataFrame, record: dict[str, Any]) -> pd.Series:
    if ledger_df.empty:
        return pd.Series(dtype=bool)
    return (
        (ledger_df["status"] == OPEN_RESULT)
        & (ledger_df["market"].fillna("").astype(str).str.lower() == DEFAULT_MARKET)
        & (
            ledger_df["match_time"].fillna("").astype(str).str.lower()
            == str(record["match_time"]).lower()
        )
        & (
            ledger_df["team1_name"].fillna("").astype(str).str.lower()
            == str(record["team1_name"]).lower()
        )
        & (
            ledger_df["team2_name"].fillna("").astype(str).str.lower()
            == str(record["team2_name"]).lower()
        )
        & (
            ledger_df["selection"].fillna("").astype(str).str.lower()
            == str(record["selection"]).lower()
        )
        & (
            ledger_df["bookmaker"].fillna("").astype(str).str.lower()
            == str(record["bookmaker"]).lower()
        )
        & (pd.to_numeric(ledger_df["odds"], errors="coerce").round(4) == round(float(record["odds"]), 4))
    )


def record_recommendations_to_ledger(
    scored_df: pd.DataFrame,
    *,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    default_stake: float = 1.0,
    notes_prefix: str = "Imported from score_bet365_value.py",
) -> dict[str, Any]:
    """Append recommended moneyline bets to the ledger, skipping duplicates."""
    if default_stake <= 0.0:
        raise ValueError("default_stake must be greater than 0.0")

    if scored_df.empty or "recommend_bet" not in scored_df.columns:
        return {"added": [], "skipped": [], "ledger_path": str(ledger_path)}

    ledger_df = load_betting_ledger(ledger_path)
    added: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    candidate_df = scored_df.loc[
        scored_df["recommend_bet"].fillna(False) & scored_df["score_available"].fillna(False)
    ].copy()
    for row in candidate_df.to_dict(orient="records"):
        stake = row.get("stake")
        normalized_stake = (
            float(stake)
            if stake is not None and not pd.isna(stake) and float(stake) > 0.0
            else float(default_stake)
        )
        notes = " | ".join(
            part
            for part in (
                notes_prefix.strip(),
                _string_or_empty(row.get("notes")),
                _string_or_empty(row.get("recommendation_reason")),
            )
            if part
        )
        record = create_bet_record(
            match_time=str(row["match_time"]),
            league_code=_string_or_empty(row.get("league_code"))
            or _string_or_empty(row.get("league")),
            team1_name=str(row["team1_name"]),
            team2_name=str(row["team2_name"]),
            market=DEFAULT_MARKET,
            selection=str(row["recommended_selection"]),
            odds=float(row["recommended_odds"]),
            stake=normalized_stake,
            fair_odds=float(row["recommended_fair_odds"]),
            model_probability=float(row["recommended_model_probability"]),
            bookmaker=_string_or_empty(row.get("bookmaker")) or DEFAULT_BOOKMAKER,
            model_name=_string_or_empty(row.get("model_name")),
            model_run_id=_string_or_empty(row.get("model_run_id")),
            snapshot_id=_string_or_empty(row.get("snapshot_id")),
            notes=notes,
        )
        duplicate_mask = _ledger_duplicate_mask(ledger_df, record)
        if not duplicate_mask.empty and bool(duplicate_mask.any()):
            skipped.append(
                {
                    "selection": record["selection"],
                    "match_time": record["match_time"],
                    "reason": "Duplicate open bet already exists.",
                }
            )
            continue

        if ledger_df.empty:
            ledger_df = pd.DataFrame([record])
        else:
            ledger_df = pd.concat([ledger_df, pd.DataFrame([record])], ignore_index=True)
        added.append(record)

    if added:
        save_betting_ledger(ledger_df, ledger_path)

    return {
        "added": added,
        "skipped": skipped,
        "ledger_path": str(ledger_path),
    }
