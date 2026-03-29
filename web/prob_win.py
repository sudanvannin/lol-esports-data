"""Web helpers for rendering model probabilities on upcoming matches."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
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


def _cache_key_for_path(path: Path) -> tuple[str, int]:
    resolved = path.resolve()
    try:
        mtime_ns = resolved.stat().st_mtime_ns
    except OSError:
        mtime_ns = -1
    return str(resolved), int(mtime_ns)


@lru_cache(maxsize=4)
def _winner_scorer_state_cached(_cache_key: tuple[str, int]) -> tuple[Any | None, str | None]:
    try:
        from src.ml.model_registry import DEFAULT_MODEL_REGISTRY_PATH
        from src.ml.winner_scorer import build_winner_scorer

        del DEFAULT_MODEL_REGISTRY_PATH
        return build_winner_scorer(), None
    except Exception as exc:
        return None, _format_model_error(exc)


def _winner_scorer_state() -> tuple[Any | None, str | None]:
    try:
        from src.ml.model_registry import DEFAULT_MODEL_REGISTRY_PATH
    except Exception:
        return _winner_scorer_state_cached(("", -1))
    return _winner_scorer_state_cached(_cache_key_for_path(DEFAULT_MODEL_REGISTRY_PATH))


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


def _build_edge_signal(
    *,
    title: str,
    diff: float,
    scale: float,
    team1_name: str,
    team2_name: str,
    note: str,
    value_suffix: str,
) -> dict[str, Any]:
    strength_score = abs(diff) / scale if scale else 0.0
    if strength_score >= 1.75:
        strength_label = "Strong"
    elif strength_score >= 1.0:
        strength_label = "Medium"
    else:
        strength_label = "Light"
    favored_name = team1_name if diff >= 0 else team2_name
    return {
        "title": title,
        "favored_name": favored_name,
        "strength_label": strength_label,
        "value_text": f"{abs(diff):.1f}{value_suffix}",
        "score": strength_score,
        "note": note,
    }


def _confidence_tier(confidence_gap_pct: float, disagreement_pct: float | None) -> tuple[str, str]:
    spread = float(disagreement_pct or 0.0)
    if confidence_gap_pct >= 28.0 and spread <= 4.0:
        return (
            "High conviction",
            "The model gap is wide and the underlying ensemble is tightly aligned.",
        )
    if confidence_gap_pct >= 16.0 and spread <= 7.5:
        return (
            "Solid edge",
            "There is a clear favorite, with only moderate disagreement across signals.",
        )
    if confidence_gap_pct <= 8.0 or spread >= 10.0:
        return (
            "Fragile read",
            "The matchup is close or the submodels are pulling in different directions.",
        )
    return (
        "Leaning favorite",
        "The favorite is credible, but the setup still carries meaningful volatility.",
    )


def build_match_explainability(match_row: dict[str, Any]) -> dict[str, Any]:
    """Explain the winner model output using the same matchup inputs it scores on."""
    context = build_match_context(match_row)
    team1 = _clean_text(match_row.get("team1"))
    team2 = _clean_text(match_row.get("team2"))
    scorer, scorer_error = _winner_scorer_state()

    if scorer is None:
        return {
            "available": False,
            "error": scorer_error or "Winner model unavailable.",
        }

    try:
        event_time = context["match_time"]
        team1_key, team1_name = scorer.resolve_team(team1)
        team2_key, team2_name = scorer.resolve_team(team2)
        feature_row, warnings, _ = scorer._build_feature_row(
            event_time=event_time,
            team1_key=team1_key,
            team1_name=team1_name,
            team2_key=team2_key,
            team2_name=team2_name,
            league_code=context["league_code"],
            split_name=context["split_name"],
            patch_version=context["patch_version"],
            best_of=context["best_of"],
            playoffs=context["playoffs"],
        )
        quote = scorer.score_match(
            team1=team1,
            team2=team2,
            match_time=context["match_time_iso"],
            league_code=context["league_code"],
            split_name=context["split_name"],
            patch_version=context["patch_version"],
            best_of=context["best_of"],
            playoffs=context["playoffs"],
        )
    except Exception as exc:
        return {
            "available": False,
            "error": _format_model_error(exc),
        }

    team1_win_pct = _probability_to_pct(quote.team1_win_prob)
    team2_win_pct = _probability_to_pct(quote.team2_win_prob)
    confidence_gap_pct = abs(team1_win_pct - team2_win_pct)
    favorite_name = team1_name if team1_win_pct >= team2_win_pct else team2_name
    disagreement_raw = getattr(quote, "model_disagreement_score", None)
    disagreement_pct = (
        round(float(disagreement_raw) * 100.0, 1)
        if disagreement_raw is not None
        else None
    )
    confidence_label, confidence_note = _confidence_tier(
        confidence_gap_pct,
        disagreement_pct,
    )

    signals: list[dict[str, Any]] = []
    team1_elo_prob = feature_row.get("team1_elo_win_prob")
    if team1_elo_prob is not None:
        signals.append(
            _build_edge_signal(
                title="Base rating edge",
                diff=(float(team1_elo_prob) - 0.5) * 100.0,
                scale=8.0,
                team1_name=team1_name,
                team2_name=team2_name,
                note="Core historical strength before matchup-specific adjustments.",
                value_suffix="pp",
            )
        )

    recent_sample = float(feature_row.get("team1_recent5_series_count", 0) or 0) + float(
        feature_row.get("team2_recent5_series_count", 0) or 0
    )
    if recent_sample >= 4:
        signals.append(
            _build_edge_signal(
                title="Recent form",
                diff=float(feature_row.get("recent5_series_win_rate_diff", 0.0)) * 100.0,
                scale=10.0,
                team1_name=team1_name,
                team2_name=team2_name,
                note="Last-five-series win-rate gap inside the core matchup history.",
                value_suffix="pp",
            )
        )

    long_run_sample = float(feature_row.get("team1_prior_series_count", 0) or 0) + float(
        feature_row.get("team2_prior_series_count", 0) or 0
    )
    if long_run_sample >= 12:
        signals.append(
            _build_edge_signal(
                title="Long-run form",
                diff=float(feature_row.get("prior_series_win_rate_diff", 0.0)) * 100.0,
                scale=8.0,
                team1_name=team1_name,
                team2_name=team2_name,
                note="Overall prior series win-rate gap from the training history.",
                value_suffix="pp",
            )
        )

    patch_sample = float(feature_row.get("team1_patch_prior_series_count", 0) or 0) + float(
        feature_row.get("team2_patch_prior_series_count", 0) or 0
    )
    if patch_sample >= 4:
        signals.append(
            _build_edge_signal(
                title="Patch fit",
                diff=float(feature_row.get("patch_prior_win_rate_diff", 0.0)) * 100.0,
                scale=8.0,
                team1_name=team1_name,
                team2_name=team2_name,
                note="How each team has performed on the current patch.",
                value_suffix="pp",
            )
        )

    split_sample = float(feature_row.get("team1_split_prior_series_count", 0) or 0) + float(
        feature_row.get("team2_split_prior_series_count", 0) or 0
    )
    if split_sample >= 4:
        signals.append(
            _build_edge_signal(
                title="Split fit",
                diff=float(feature_row.get("split_prior_win_rate_diff", 0.0)) * 100.0,
                scale=8.0,
                team1_name=team1_name,
                team2_name=team2_name,
                note="Relative performance in this split or seasonal context.",
                value_suffix="pp",
            )
        )

    h2h_count = int(feature_row.get("h2h_prior_series_count", 0) or 0)
    if h2h_count >= 2:
        signals.append(
            _build_edge_signal(
                title="Head-to-head",
                diff=(float(feature_row.get("h2h_team1_series_win_rate", 0.5)) - 0.5)
                * 100.0,
                scale=10.0,
                team1_name=team1_name,
                team2_name=team2_name,
                note="Direct prior series results between these two teams.",
                value_suffix="pp",
            )
        )

    continuity_diff = float(feature_row.get("recent3_avg_roster_overlap_diff", 0.0) or 0.0)
    if abs(continuity_diff) > 0.01:
        signals.append(
            _build_edge_signal(
                title="Roster continuity",
                diff=continuity_diff,
                scale=1.0,
                team1_name=team1_name,
                team2_name=team2_name,
                note="Average overlap with the previous three series rosters.",
                value_suffix=" players",
            )
        )

    churn_diff = -float(feature_row.get("roster_new_player_count_diff", 0.0) or 0.0)
    if abs(churn_diff) > 0.01:
        signals.append(
            _build_edge_signal(
                title="Roster stability",
                diff=churn_diff,
                scale=1.0,
                team1_name=team1_name,
                team2_name=team2_name,
                note="Teams with fewer new players generally carry less integration risk.",
                value_suffix=" players",
            )
        )

    signals = sorted(signals, key=lambda item: item["score"], reverse=True)[:4]

    individual_model_probs = getattr(quote, "individual_model_probs", {}) or {}
    model_rows = [
        {
            "model_name": model_name.replace("_", " ").title(),
            "team1_win_pct": _probability_to_pct(probability),
        }
        for model_name, probability in sorted(individual_model_probs.items())
    ]
    if disagreement_pct is None:
        agreement_label = "Single-model output"
        agreement_note = "This deploy is serving one winner model instead of an ensemble."
    elif disagreement_pct <= 3.0:
        agreement_label = "Tight agreement"
        agreement_note = "The ensemble models are tightly clustered around the same read."
    elif disagreement_pct <= 6.5:
        agreement_label = "Moderate spread"
        agreement_note = "The ensemble agrees on direction, but not on exact confidence."
    else:
        agreement_label = "High disagreement"
        agreement_note = "Submodels disagree materially, so the read is more fragile."

    sample_context = [
        {
            "label": "Core series",
            "team1_value": int(feature_row.get("team1_prior_series_count", 0) or 0),
            "team2_value": int(feature_row.get("team2_prior_series_count", 0) or 0),
        },
        {
            "label": "Recent 5 sample",
            "team1_value": int(feature_row.get("team1_recent5_series_count", 0) or 0),
            "team2_value": int(feature_row.get("team2_recent5_series_count", 0) or 0),
        },
        {
            "label": "Patch sample",
            "team1_value": int(feature_row.get("team1_patch_prior_series_count", 0) or 0),
            "team2_value": int(feature_row.get("team2_patch_prior_series_count", 0) or 0),
        },
        {
            "label": "Split sample",
            "team1_value": int(feature_row.get("team1_split_prior_series_count", 0) or 0),
            "team2_value": int(feature_row.get("team2_split_prior_series_count", 0) or 0),
        },
    ]

    lead_signals = ", ".join(item["title"].lower() for item in signals[:2]) if signals else ""
    summary = (
        f"{favorite_name} projects as the favorite mainly through {lead_signals}."
        if lead_signals
        else f"{favorite_name} projects as the favorite on the current prematch inputs."
    )

    return {
        "available": True,
        "favorite_name": favorite_name,
        "favorite_prob_pct": max(team1_win_pct, team2_win_pct),
        "confidence_gap_pct": round(confidence_gap_pct, 1),
        "confidence_label": confidence_label,
        "confidence_note": confidence_note,
        "summary": summary,
        "signals": signals,
        "sample_context": sample_context,
        "model_agreement": {
            "label": agreement_label,
            "note": agreement_note,
            "disagreement_pct": disagreement_pct,
            "calibration_method": getattr(quote, "calibration_method", "single_model"),
            "models": model_rows,
        },
        "warnings": list(dict.fromkeys(list(getattr(quote, "warnings", [])) + list(warnings))),
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
