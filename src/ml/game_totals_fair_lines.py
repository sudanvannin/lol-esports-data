"""Prematch fair lines for single-game totals markets."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from .fair_odds import DEFAULT_LATEST_RUN_POINTER, PrematchFairOddsScorer, _probability_to_fair_odds
from .game_totals_baseline import (
    BASELINE_MIN_LEAGUE_SAMPLES,
    BASELINE_MIN_PATCH_SAMPLES,
    MARKET_TO_TARGET,
    TARGET_TO_MARKET,
    build_game_totals_pairs,
    build_game_totals_team_history_frame,
    build_upcoming_game_totals_feature_row,
)

DEFAULT_TOTALS_LINES = {
    "total_kills": 29.5,
    "total_dragons": 4.5,
    "total_towers": 12.5,
    "total_barons": 1.5,
    "total_inhibitors": 1.5,
}
RECENT_TEAM_GAMES = 30
BASELINE_GAMES = 400
TEAM_PRIOR_WEIGHT = 8.0
BASELINE_RECENCY_HALFLIFE_DAYS = 120.0
TEAM_RECENCY_HALFLIFE_DAYS = 90.0
DEFAULT_TOTALS_MODELS_POINTER = Path("data/models/game_totals_baseline/latest_run.json")


@dataclass(slots=True)
class TotalsMarketQuote:
    """Fair lines for a single totals market."""

    market: str
    line: float
    predicted_mean: float
    predicted_variance: float
    distribution: str
    over_prob: float
    under_prob: float
    over_fair_odds: float
    under_fair_odds: float
    team1_sample: int
    team2_sample: int
    baseline_sample: int


@dataclass(slots=True)
class TotalsFairLinesQuote:
    """Serializable output for totals market scoring."""

    snapshot_id: str
    match_time: str
    league_code: str
    patch_version: str
    team1_key: str
    team1_name: str
    team2_key: str
    team2_name: str
    warnings: list[str]
    markets: list[TotalsMarketQuote]


def _clip_probability(probability: float) -> float:
    return float(min(max(probability, 1e-6), 1.0 - 1e-6))


def _weighted_mean_variance(
    values: pd.Series,
    event_times: pd.Series,
    reference_time: pd.Timestamp,
    halflife_days: float,
) -> tuple[float, float, int]:
    clean = pd.DataFrame({"value": values, "event_time": event_times}).dropna()
    if clean.empty:
        return 0.0, 0.0, 0

    age_days = (
        reference_time - pd.to_datetime(clean["event_time"], utc=True, errors="coerce")
    ).dt.total_seconds() / 86400.0
    weights = np.exp(-np.maximum(age_days, 0.0) / halflife_days)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return 0.0, 0.0, 0

    values_array = clean["value"].astype(float).to_numpy()
    mean_value = float(np.average(values_array, weights=weights))
    variance_value = float(np.average((values_array - mean_value) ** 2, weights=weights))
    return mean_value, variance_value, int(len(clean))


def _shrunk_mean(observed_mean: float, observed_n: int, prior_mean: float, prior_weight: float) -> float:
    return (observed_mean * observed_n + prior_mean * prior_weight) / (observed_n + prior_weight)


def _shrunk_variance(
    observed_variance: float,
    observed_n: int,
    prior_variance: float,
    prior_weight: float,
) -> float:
    return (observed_variance * observed_n + prior_variance * prior_weight) / (observed_n + prior_weight)


def estimate_discrete_market(
    mean_value: float,
    variance_value: float,
    line: float,
) -> tuple[str, float, float]:
    """Estimate fair over/under probabilities for a discrete total."""
    threshold = math.floor(line)
    mean_value = max(float(mean_value), 1e-6)
    variance_value = max(float(variance_value), mean_value)

    if variance_value > mean_value + 1e-6:
        size = mean_value * mean_value / max(variance_value - mean_value, 1e-6)
        probability = size / (size + mean_value)
        under_prob = float(nbinom.cdf(threshold, size, probability))
        distribution = "negative_binomial"
    else:
        under_prob = float(poisson.cdf(threshold, mean_value))
        distribution = "poisson"

    under_prob = _clip_probability(under_prob)
    over_prob = _clip_probability(1.0 - under_prob)
    return distribution, over_prob, under_prob


class PrematchGameTotalsScorer:
    """Score single-game totals markets for a future matchup."""

    def __init__(
        self,
        run_pointer: Path = DEFAULT_LATEST_RUN_POINTER,
        totals_models_pointer: Path = DEFAULT_TOTALS_MODELS_POINTER,
    ) -> None:
        self.team_resolver = PrematchFairOddsScorer(run_pointer=run_pointer)
        self.snapshot_id = self.team_resolver.snapshot_id
        self.snapshot_dir = self.team_resolver.snapshot_dir
        self.core_leagues = set(self.team_resolver.seen_leagues)

        self.games_df = build_game_totals_pairs(self.snapshot_dir)
        self.team_history_df = build_game_totals_team_history_frame(self.games_df)
        self.game_totals_df = self.games_df.rename(columns=TARGET_TO_MARKET)
        self.market_models = self._load_market_models(totals_models_pointer)

    @staticmethod
    def _load_market_models(pointer_path: Path) -> dict[str, dict[str, Any]]:
        if not pointer_path.exists():
            return {}

        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
        if "targets" not in payload:
            best_models_path = Path(payload.get("best_models_path", ""))
            if not best_models_path.exists():
                return {}
            payload = json.loads(best_models_path.read_text(encoding="utf-8"))

        market_models: dict[str, dict[str, Any]] = {}
        for target_name, target_payload in payload.get("targets", {}).items():
            model_path = Path(target_payload["model_path"])
            if not model_path.exists():
                continue
            with model_path.open("rb") as file_obj:
                model = pickle.load(file_obj)
            market_models[str(target_payload["market"])] = {
                "target": str(target_name),
                "model": model,
                "model_name": str(target_payload["model_name"]),
                "dispersion_alpha": float(target_payload["dispersion_alpha"]),
            }
        return market_models

    def _recent_team_games(
        self,
        team_key: str,
        event_time: pd.Timestamp,
    ) -> pd.DataFrame:
        return (
            self.team_history_df.loc[
                (self.team_history_df["team_key"] == team_key) & (self.team_history_df["event_time"] < event_time)
            ]
            .sort_values(["event_time", "game_id"], kind="stable")
            .tail(RECENT_TEAM_GAMES)
            .reset_index(drop=True)
        )

    def _baseline_games(
        self,
        event_time: pd.Timestamp,
        league_code: str,
        patch_version: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        warnings: list[str] = []
        eligible = self.game_totals_df.loc[self.game_totals_df["event_time"] < event_time].copy()

        patch_frame = eligible.loc[eligible["patch_version"] == patch_version]
        if len(patch_frame) >= BASELINE_MIN_PATCH_SAMPLES:
            return patch_frame.sort_values(["event_time", "game_id"], kind="stable").tail(BASELINE_GAMES), warnings
        warnings.append(f"Patch '{patch_version}' ainda sem amostra suficiente; usando baseline mais amplo.")

        league_frame = eligible.loc[eligible["league_code"] == league_code]
        if len(league_frame) >= BASELINE_MIN_LEAGUE_SAMPLES:
            return league_frame.sort_values(["event_time", "game_id"], kind="stable").tail(BASELINE_GAMES), warnings
        if league_code not in self.core_leagues:
            warnings.append(f"Liga {league_code} ainda sem amostra suficiente; usando baseline cross-league.")

        core_frame = eligible.loc[eligible["league_code"].isin(self.core_leagues)]
        return core_frame.sort_values(["event_time", "game_id"], kind="stable").tail(BASELINE_GAMES), warnings

    def _score_market_heuristic(
        self,
        metric: str,
        line: float,
        team1_games: pd.DataFrame,
        team2_games: pd.DataFrame,
        baseline_games: pd.DataFrame,
        reference_time: pd.Timestamp,
    ) -> TotalsMarketQuote:
        baseline_mean, baseline_var, baseline_n = _weighted_mean_variance(
            baseline_games[metric],
            baseline_games["event_time"],
            reference_time=reference_time,
            halflife_days=BASELINE_RECENCY_HALFLIFE_DAYS,
        )
        team1_mean, team1_var, team1_n = _weighted_mean_variance(
            team1_games[metric],
            team1_games["event_time"],
            reference_time=reference_time,
            halflife_days=TEAM_RECENCY_HALFLIFE_DAYS,
        )
        team2_mean, team2_var, team2_n = _weighted_mean_variance(
            team2_games[metric],
            team2_games["event_time"],
            reference_time=reference_time,
            halflife_days=TEAM_RECENCY_HALFLIFE_DAYS,
        )

        if baseline_n == 0:
            fallback_values = pd.concat([team1_games[metric], team2_games[metric]], ignore_index=True)
            baseline_mean = float(fallback_values.mean())
            baseline_var = float(fallback_values.var(ddof=0))

        team1_mean = _shrunk_mean(team1_mean, team1_n, baseline_mean, TEAM_PRIOR_WEIGHT)
        team2_mean = _shrunk_mean(team2_mean, team2_n, baseline_mean, TEAM_PRIOR_WEIGHT)
        team1_var = _shrunk_variance(team1_var, team1_n, baseline_var, TEAM_PRIOR_WEIGHT)
        team2_var = _shrunk_variance(team2_var, team2_n, baseline_var, TEAM_PRIOR_WEIGHT)

        predicted_mean = float((team1_mean + team2_mean) / 2.0)
        predicted_variance = float(max((team1_var + team2_var) / 2.0, predicted_mean))
        distribution, over_prob, under_prob = estimate_discrete_market(
            mean_value=predicted_mean,
            variance_value=predicted_variance,
            line=line,
        )

        return TotalsMarketQuote(
            market=metric,
            line=float(line),
            predicted_mean=predicted_mean,
            predicted_variance=predicted_variance,
            distribution=distribution,
            over_prob=over_prob,
            under_prob=under_prob,
            over_fair_odds=_probability_to_fair_odds(over_prob),
            under_fair_odds=_probability_to_fair_odds(under_prob),
            team1_sample=int(team1_n),
            team2_sample=int(team2_n),
            baseline_sample=int(baseline_n),
        )

    def _score_market_model(
        self,
        metric: str,
        line: float,
        feature_row: pd.DataFrame,
        feature_meta: dict[str, Any],
    ) -> TotalsMarketQuote:
        market_model = self.market_models[metric]
        predicted_mean = float(market_model["model"].predict(feature_row)[0])
        predicted_mean = max(predicted_mean, 1e-6)
        alpha = float(market_model["dispersion_alpha"])
        predicted_variance = max(predicted_mean + alpha * predicted_mean * predicted_mean, predicted_mean)
        distribution, over_prob, under_prob = estimate_discrete_market(
            mean_value=predicted_mean,
            variance_value=predicted_variance,
            line=line,
        )

        return TotalsMarketQuote(
            market=metric,
            line=float(line),
            predicted_mean=predicted_mean,
            predicted_variance=float(predicted_variance),
            distribution=distribution,
            over_prob=over_prob,
            under_prob=under_prob,
            over_fair_odds=_probability_to_fair_odds(over_prob),
            under_fair_odds=_probability_to_fair_odds(under_prob),
            team1_sample=int(feature_meta["team1_prior_game_count"]),
            team2_sample=int(feature_meta["team2_prior_game_count"]),
            baseline_sample=int(feature_meta["baseline_sample_count"]),
        )

    def score_match(
        self,
        team1: str,
        team2: str,
        match_time: str,
        league_code: str,
        patch_version: str,
        market_lines: dict[str, float] | None = None,
        split_name: str | None = None,
        playoffs: bool = False,
    ) -> TotalsFairLinesQuote:
        event_time = pd.Timestamp(match_time)
        if event_time.tzinfo is None:
            event_time = event_time.tz_localize("UTC")
        else:
            event_time = event_time.tz_convert("UTC")

        resolved_pairs = [
            self.team_resolver.resolve_team(team1),
            self.team_resolver.resolve_team(team2),
        ]
        resolved_pairs = sorted(resolved_pairs, key=lambda item: (item[0], item[1]))
        team1_key, team1_name = resolved_pairs[0]
        team2_key, team2_name = resolved_pairs[1]

        lines = DEFAULT_TOTALS_LINES if market_lines is None else market_lines
        warnings: list[str] = []
        feature_row, feature_meta = build_upcoming_game_totals_feature_row(
            self.games_df,
            self.team_history_df,
            snapshot_id=self.snapshot_id,
            team1_key=team1_key,
            team1_name=team1_name,
            team2_key=team2_key,
            team2_name=team2_name,
            event_time=event_time,
            league_code=league_code,
            split_name=split_name or league_code,
            patch_version=patch_version,
            playoffs=playoffs,
            game_id=f"upcoming_{league_code}_{event_time.strftime('%Y%m%dT%H%M%SZ')}",
        )

        if feature_meta["patch_sample_count"] < BASELINE_MIN_PATCH_SAMPLES:
            warnings.append(f"Patch '{patch_version}' ainda sem amostra suficiente; usando baseline mais amplo.")
        if feature_meta["baseline_source"] == "global" and feature_meta["league_sample_count"] < BASELINE_MIN_LEAGUE_SAMPLES:
            if league_code not in self.core_leagues:
                warnings.append(f"Liga {league_code} ainda sem amostra suficiente; usando baseline cross-league.")
        if feature_meta["team1_prior_game_count"] < 10:
            warnings.append(f"{team1_name} tem pouca amostra recente de mapas ({feature_meta['team1_prior_game_count']}).")
        if feature_meta["team2_prior_game_count"] < 10:
            warnings.append(f"{team2_name} tem pouca amostra recente de mapas ({feature_meta['team2_prior_game_count']}).")

        force_heuristic = False
        if not self.market_models:
            warnings.append("Modelos treinados de totals nao encontrados; usando fallback heuristico.")
            force_heuristic = True

        team1_games = pd.DataFrame()
        team2_games = pd.DataFrame()
        baseline_games = pd.DataFrame()
        heuristic_warnings: list[str] = []

        markets: list[TotalsMarketQuote] = []
        for metric, line in lines.items():
            if not force_heuristic and metric in self.market_models and MARKET_TO_TARGET.get(metric):
                markets.append(
                    self._score_market_model(
                        metric=metric,
                        line=float(line),
                        feature_row=feature_row,
                        feature_meta=feature_meta,
                    )
                )
                continue

            if not force_heuristic and metric not in self.market_models:
                warnings.append(f"Modelo treinado ausente para {metric}; usando fallback heuristico nesse mercado.")
            if team1_games.empty and team2_games.empty and baseline_games.empty:
                team1_games = self._recent_team_games(team1_key, event_time)
                team2_games = self._recent_team_games(team2_key, event_time)
                baseline_games, heuristic_warnings = self._baseline_games(
                    event_time,
                    league_code=league_code,
                    patch_version=patch_version,
                )
            markets.append(
                self._score_market_heuristic(
                    metric=metric,
                    line=float(line),
                    team1_games=team1_games,
                    team2_games=team2_games,
                    baseline_games=baseline_games,
                    reference_time=event_time,
                )
            )

        warnings.extend(item for item in heuristic_warnings if item not in warnings)
        return TotalsFairLinesQuote(
            snapshot_id=self.snapshot_id,
            match_time=event_time.isoformat(),
            league_code=league_code,
            patch_version=patch_version,
            team1_key=team1_key,
            team1_name=team1_name,
            team2_key=team2_key,
            team2_name=team2_name,
            warnings=warnings,
            markets=markets,
        )
