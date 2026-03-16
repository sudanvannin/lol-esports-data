"""Web helpers for rendering model probabilities on upcoming matches."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from typing import Any

import pandas as pd

LEAGUE_CODE_ALIASES = {
    "cblol": "CBLOL",
    "first stand": "FST",
    "first stand tournament": "FST",
    "lck": "LCK",
    "lck challengers": "LCKC",
    "lcs": "LCS",
    "lec": "LEC",
    "lpl": "LPL",
    "lta": "LTA",
    "lta north": "LTA N",
    "lta south": "LTA S",
    "mid-season invitational": "MSI",
    "msi": "MSI",
    "world championship": "WLDs",
    "worlds": "WLDs",
}
MARKET_LABELS = {
    "total_kills": "Total Kills",
    "total_dragons": "Total Dragons",
    "total_towers": "Total Towers",
    "total_barons": "Total Barons",
    "total_inhibitors": "Total Inhibitors",
}
PLAYOFF_KEYWORDS = (
    "bracket",
    "elimination",
    "final",
    "gauntlet",
    "knockout",
    "playoff",
    "quarter",
    "semi",
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _safe_match_time(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _safe_best_of(value: Any) -> int:
    best_of = pd.to_numeric(value, errors="coerce")
    if pd.isna(best_of):
        return 3
    return max(int(best_of), 1)


def _probability_to_pct(value: float) -> float:
    return round(float(value) * 100.0, 1)


def _format_model_error(exc: Exception) -> str:
    message = _clean_text(exc)
    lowered = message.lower()
    if isinstance(exc, (FileNotFoundError, ModuleNotFoundError, ImportError)):
        return "Model package is not available on this deploy."
    if "no such file" in lowered or "cannot find" in lowered:
        return "Model artifacts are missing on this deploy."
    return message or exc.__class__.__name__


@lru_cache(maxsize=1)
def _winner_scorer_state() -> tuple[Any | None, str | None]:
    try:
        from src.ml.fair_odds import PrematchFairOddsScorer

        return PrematchFairOddsScorer(), None
    except Exception as exc:
        return None, _format_model_error(exc)


@lru_cache(maxsize=1)
def _totals_scorer_state() -> tuple[Any | None, str | None]:
    try:
        from src.ml.game_totals_fair_lines import PrematchGameTotalsScorer

        return PrematchGameTotalsScorer(), None
    except Exception as exc:
        return None, _format_model_error(exc)


def _infer_league_code(match_row: dict[str, Any]) -> str:
    for candidate in (
        _clean_text(match_row.get("league")),
        _clean_text(match_row.get("league_label")),
        _clean_text(match_row.get("event_name")),
    ):
        slug = _slugify(candidate)
        if slug in LEAGUE_CODE_ALIASES:
            return LEAGUE_CODE_ALIASES[slug]

        upper = candidate.upper()
        if upper in {
            "CBLOL",
            "FST",
            "LCK",
            "LCKC",
            "LCS",
            "LEC",
            "LPL",
            "LTA",
            "LTA N",
            "LTA S",
            "MSI",
            "WLDS",
        }:
            return "WLDs" if upper == "WLDS" else upper

    return _clean_text(match_row.get("league")) or "UNKNOWN"


def _infer_split_name(match_row: dict[str, Any], league_code: str) -> str:
    combined = " ".join(
        part
        for part in (
            _clean_text(match_row.get("event_name")),
            _clean_text(match_row.get("phase_label")),
            _clean_text(match_row.get("league")),
        )
        if part
    )
    lowered = combined.lower()

    for split_name in ("Winter", "Spring", "Summer"):
        if split_name.lower() in lowered:
            return split_name

    split_match = re.search(r"split\s+(\d)", lowered)
    if split_match:
        return f"Split {split_match.group(1)}"

    if "season finals" in lowered:
        return "Season Finals"
    if league_code == "MSI":
        return "MSI"
    if league_code == "WLDs":
        return "WLDs"
    if league_code == "FST":
        return "FST"
    return league_code


def _infer_patch_version(match_row: dict[str, Any]) -> str:
    patch = _clean_text(match_row.get("patch"))
    if patch:
        return patch

    scorer, _ = _winner_scorer_state()
    if scorer is not None:
        patches = sorted(
            _clean_text(item)
            for item in getattr(scorer, "seen_patches", set())
            if _clean_text(item)
        )
        if patches:
            return patches[-1]

    return "unknown"


def _infer_playoffs(match_row: dict[str, Any]) -> bool:
    text = " ".join(
        (
            _clean_text(match_row.get("event_name")),
            _clean_text(match_row.get("phase_label")),
        )
    ).lower()
    return any(keyword in text for keyword in PLAYOFF_KEYWORDS)


def build_match_context(match_row: dict[str, Any]) -> dict[str, Any]:
    league_code = _infer_league_code(match_row)
    match_time = _safe_match_time(match_row.get("match_time"))
    return {
        "match_id": _clean_text(match_row.get("match_id")),
        "league_code": league_code,
        "split_name": _infer_split_name(match_row, league_code),
        "patch_version": _infer_patch_version(match_row),
        "best_of": _safe_best_of(match_row.get("best_of")),
        "playoffs": _infer_playoffs(match_row),
        "match_time": match_time,
        "match_time_iso": match_time.isoformat(),
    }


@lru_cache(maxsize=256)
def _score_winner_cached(
    match_id: str,
    team1: str,
    team2: str,
    match_time_iso: str,
    league_code: str,
    split_name: str,
    patch_version: str,
    best_of: int,
    playoffs: bool,
) -> dict[str, Any]:
    del match_id
    scorer, scorer_error = _winner_scorer_state()
    if scorer is None:
        return {
            "available": False,
            "error": scorer_error or "Winner model unavailable.",
            "warnings": [],
        }

    try:
        quote = scorer.score_match(
            team1=team1,
            team2=team2,
            match_time=match_time_iso,
            league_code=league_code,
            split_name=split_name,
            patch_version=patch_version,
            best_of=int(best_of),
            playoffs=bool(playoffs),
        )
    except Exception as exc:
        return {
            "available": False,
            "error": _format_model_error(exc),
            "warnings": [],
        }

    favorite_name = (
        quote.team1_name
        if quote.team1_win_prob >= quote.team2_win_prob
        else quote.team2_name
    )
    favorite_prob = max(quote.team1_win_prob, quote.team2_win_prob)
    return {
        "available": True,
        "team1_name": quote.team1_name,
        "team2_name": quote.team2_name,
        "team1_win_prob": float(quote.team1_win_prob),
        "team2_win_prob": float(quote.team2_win_prob),
        "team1_win_pct": _probability_to_pct(quote.team1_win_prob),
        "team2_win_pct": _probability_to_pct(quote.team2_win_prob),
        "team1_fair_odds": float(quote.team1_fair_odds),
        "team2_fair_odds": float(quote.team2_fair_odds),
        "favorite_name": favorite_name,
        "favorite_prob_pct": _probability_to_pct(favorite_prob),
        "warnings": list(quote.warnings),
    }


@lru_cache(maxsize=256)
def _score_totals_cached(
    match_id: str,
    team1: str,
    team2: str,
    match_time_iso: str,
    league_code: str,
    split_name: str,
    patch_version: str,
    playoffs: bool,
) -> dict[str, Any]:
    del match_id
    scorer, scorer_error = _totals_scorer_state()
    if scorer is None:
        return {
            "available": False,
            "error": scorer_error or "Totals model unavailable.",
            "warnings": [],
            "markets": [],
        }

    try:
        quote = scorer.score_match(
            team1=team1,
            team2=team2,
            match_time=match_time_iso,
            league_code=league_code,
            split_name=split_name,
            patch_version=patch_version,
            playoffs=bool(playoffs),
        )
    except Exception as exc:
        return {
            "available": False,
            "error": _format_model_error(exc),
            "warnings": [],
            "markets": [],
        }

    markets: list[dict[str, Any]] = []
    for market in quote.markets:
        markets.append(
            {
                "market_key": market.market,
                "market_label": MARKET_LABELS.get(market.market, market.market),
                "line": float(market.line),
                "predicted_mean": round(float(market.predicted_mean), 2),
                "distribution": market.distribution.replace("_", " "),
                "over_prob": float(market.over_prob),
                "under_prob": float(market.under_prob),
                "over_prob_pct": _probability_to_pct(market.over_prob),
                "under_prob_pct": _probability_to_pct(market.under_prob),
                "over_fair_odds": float(market.over_fair_odds),
                "under_fair_odds": float(market.under_fair_odds),
                "team1_sample": int(market.team1_sample),
                "team2_sample": int(market.team2_sample),
                "baseline_sample": int(market.baseline_sample),
            }
        )

    return {
        "available": True,
        "warnings": list(quote.warnings),
        "markets": markets,
    }


def build_prob_win_board(
    rows: list[dict[str, Any]],
    preview_limit: int = 8,
) -> list[dict[str, Any]]:
    board: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        context = build_match_context(row)
        winner = None
        if index < preview_limit:
            winner = _score_winner_cached(
                context["match_id"],
                _clean_text(row.get("team1")),
                _clean_text(row.get("team2")),
                context["match_time_iso"],
                context["league_code"],
                context["split_name"],
                context["patch_version"],
                context["best_of"],
                context["playoffs"],
            )

        card = dict(row)
        card["context"] = context
        card["winner_market"] = winner
        board.append(card)

    return board


def build_prob_win_detail(match_row: dict[str, Any]) -> dict[str, Any]:
    context = build_match_context(match_row)
    team1 = _clean_text(match_row.get("team1"))
    team2 = _clean_text(match_row.get("team2"))
    winner_market = _score_winner_cached(
        context["match_id"],
        team1,
        team2,
        context["match_time_iso"],
        context["league_code"],
        context["split_name"],
        context["patch_version"],
        context["best_of"],
        context["playoffs"],
    )
    totals_market = _score_totals_cached(
        context["match_id"],
        team1,
        team2,
        context["match_time_iso"],
        context["league_code"],
        context["split_name"],
        context["patch_version"],
        context["playoffs"],
    )

    warnings: list[str] = []
    warnings.extend(winner_market.get("warnings", []))
    warnings.extend(
        item for item in totals_market.get("warnings", []) if item not in warnings
    )
    if not winner_market.get("available") and winner_market.get("error"):
        warnings.append(winner_market["error"])
    if totals_market.get("available") is False and totals_market.get("error"):
        if totals_market["error"] not in warnings:
            warnings.append(totals_market["error"])

    return {
        "match": dict(match_row),
        "context": context,
        "winner_market": winner_market,
        "totals_market": totals_market,
        "warnings": warnings,
        "has_any_market": bool(
            winner_market.get("available") or totals_market.get("available")
        ),
        "confidence_gap_pct": (
            abs(
                _probability_to_pct(winner_market["team1_win_prob"])
                - _probability_to_pct(winner_market["team2_win_prob"])
            )
            if winner_market.get("available")
            else math.nan
        ),
    }


def flatten_prob_win_detail(
    detail: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Flatten a prob-win detail payload for database upload."""
    match_row = dict(detail["match"])
    context = dict(detail["context"])
    winner_market = dict(detail["winner_market"])
    totals_market = dict(detail["totals_market"])
    warnings = list(detail.get("warnings", []))

    match_record = {
        "match_id": _clean_text(match_row.get("match_id")),
        "match_time": match_row.get("match_time"),
        "match_date": match_row.get("match_date"),
        "league": _clean_text(match_row.get("league")),
        "league_label": _clean_text(match_row.get("league_label")),
        "event_name": _clean_text(match_row.get("event_name")),
        "phase_label": _clean_text(match_row.get("phase_label")),
        "team1": _clean_text(match_row.get("team1")),
        "team2": _clean_text(match_row.get("team2")),
        "best_of": context.get("best_of"),
        "patch": _clean_text(match_row.get("patch"))
        or _clean_text(context.get("patch_version")),
        "league_code": _clean_text(context.get("league_code")),
        "split_name": _clean_text(context.get("split_name")),
        "playoffs": bool(context.get("playoffs")),
        "winner_available": bool(winner_market.get("available")),
        "winner_error": _clean_text(winner_market.get("error")),
        "team1_win_prob": winner_market.get("team1_win_prob"),
        "team2_win_prob": winner_market.get("team2_win_prob"),
        "team1_win_pct": winner_market.get("team1_win_pct"),
        "team2_win_pct": winner_market.get("team2_win_pct"),
        "team1_fair_odds": winner_market.get("team1_fair_odds"),
        "team2_fair_odds": winner_market.get("team2_fair_odds"),
        "favorite_name": _clean_text(winner_market.get("favorite_name")),
        "favorite_prob_pct": winner_market.get("favorite_prob_pct"),
        "totals_available": bool(totals_market.get("available")),
        "totals_error": _clean_text(totals_market.get("error")),
        "warnings_json": json.dumps(warnings, ensure_ascii=True),
    }

    totals_records: list[dict[str, Any]] = []
    for market in totals_market.get("markets", []):
        totals_records.append(
            {
                "match_id": match_record["match_id"],
                "market_key": _clean_text(market.get("market_key")),
                "market_label": _clean_text(market.get("market_label")),
                "line": market.get("line"),
                "predicted_mean": market.get("predicted_mean"),
                "distribution": _clean_text(market.get("distribution")),
                "over_prob": market.get("over_prob"),
                "under_prob": market.get("under_prob"),
                "over_prob_pct": market.get("over_prob_pct"),
                "under_prob_pct": market.get("under_prob_pct"),
                "over_fair_odds": market.get("over_fair_odds"),
                "under_fair_odds": market.get("under_fair_odds"),
                "team1_sample": market.get("team1_sample"),
                "team2_sample": market.get("team2_sample"),
                "baseline_sample": market.get("baseline_sample"),
            }
        )

    return match_record, totals_records
