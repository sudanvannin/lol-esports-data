"""Temporal baseline training for single-game totals markets."""

from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .prematch_baseline import (
    DEFAULT_TEST_START_DATE,
    DEFAULT_TRAIN_END_DATE,
    DEFAULT_VALIDATION_END_DATE,
    DEFAULT_VALIDATION_START_DATE,
    create_calendar_train_validation_test_split,
)

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - environment dependent
    XGBRegressor = None


logger = logging.getLogger(__name__)

LATEST_SNAPSHOT_POINTER = Path("data/gold/latest_snapshot.json")
DEFAULT_OUTPUT_ROOT = Path("data/models/game_totals_baseline")
TARGET_COLUMNS = [
    "target_total_kills",
    "target_total_dragons",
    "target_total_towers",
    "target_total_barons",
    "target_total_inhibitors",
]
TARGET_TO_MARKET = {
    "target_total_kills": "total_kills",
    "target_total_dragons": "total_dragons",
    "target_total_towers": "total_towers",
    "target_total_barons": "total_barons",
    "target_total_inhibitors": "total_inhibitors",
}
MARKET_TO_TARGET = {market: target for target, market in TARGET_TO_MARKET.items()}
META_COLUMNS = [
    "snapshot_id",
    "game_id",
    "event_time",
    "event_date",
    "team1_name",
    "team2_name",
]
CATEGORICAL_COLUMNS = ["league_code", "split_name", "patch_version", "team1_key", "team2_key"]
MODEL_LEAGUES = ("LCK", "LEC", "LPL", "LCS", "CBLOL", "LTA", "LTA S", "LTA N", "MSI", "WLDs", "FST")
RECENT_WINDOWS = (5, 10, 30)
BASELINE_MAXLEN = 400
BASELINE_MIN_PATCH_SAMPLES = 50
BASELINE_MIN_LEAGUE_SAMPLES = 50
FEATURE_METRICS = [
    "total_kills",
    "total_dragons",
    "total_towers",
    "total_barons",
    "total_inhibitors",
    "game_length_seconds",
    "team_kills",
    "team_dragons",
    "team_towers",
    "team_barons",
    "team_inhibitors",
]


@dataclass(slots=True)
class GameTotalsRunResult:
    """Artifacts produced by the game-totals baseline training pipeline."""

    run_id: str
    output_dir: Path
    snapshot_id: str
    features_path: Path
    metrics_path: Path
    train_predictions_path: Path
    test_predictions_path: Path
    best_models_path: Path


def _utc_now() -> datetime:
    return datetime.now(UTC)


def resolve_latest_snapshot(pointer_path: Path = LATEST_SNAPSHOT_POINTER) -> tuple[str, Path]:
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    snapshot_id = pointer["snapshot_id"]
    snapshot_dir = Path(pointer["snapshot_dir"])
    return snapshot_id, snapshot_dir


def _weighted_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _weighted_var(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = _weighted_mean(values)
    return float(sum((value - mean_value) ** 2 for value in values) / len(values))


def _days_since(last_seen: pd.Timestamp | None, current_time: pd.Timestamp) -> float:
    if last_seen is None or pd.isna(last_seen):
        return float("nan")
    return (current_time - last_seen).total_seconds() / 86400.0


def _build_game_pairs(snapshot_dir: Path) -> pd.DataFrame:
    columns = [
        "game_id",
        "game_date",
        "game_datetime",
        "league_code",
        "split_name",
        "patch_version",
        "playoffs",
        "team_key",
        "team_name",
        "team_kills",
        "towers",
        "dragons",
        "barons",
        "inhibitors",
        "game_length_seconds",
    ]
    team_df = pd.read_parquet(snapshot_dir / "fact_game_team.parquet", columns=columns)
    team_df["game_date"] = pd.to_datetime(team_df["game_date"], utc=True, errors="coerce")
    team_df["game_datetime"] = pd.to_datetime(team_df["game_datetime"], utc=True, errors="coerce")
    team_df["event_time"] = team_df["game_datetime"].fillna(team_df["game_date"])
    team_df["split_name"] = team_df["split_name"].fillna(team_df["league_code"])
    team_df["team_key"] = team_df["team_key"].map(str)
    team_df["team_name"] = team_df["team_name"].fillna(team_df["team_key"]).map(str)
    team_df = team_df.loc[team_df["league_code"].isin(MODEL_LEAGUES)].copy()

    records: list[dict[str, Any]] = []
    for game_id, frame in team_df.groupby("game_id", sort=False):
        if len(frame) != 2:
            continue
        ordered = frame.sort_values(["team_key", "team_name"], kind="stable").reset_index(drop=True)
        team1 = ordered.iloc[0]
        team2 = ordered.iloc[1]
        team1_kills = 0.0 if pd.isna(team1["team_kills"]) else float(team1["team_kills"])
        team2_kills = 0.0 if pd.isna(team2["team_kills"]) else float(team2["team_kills"])
        team1_dragons = 0.0 if pd.isna(team1["dragons"]) else float(team1["dragons"])
        team2_dragons = 0.0 if pd.isna(team2["dragons"]) else float(team2["dragons"])
        team1_towers = 0.0 if pd.isna(team1["towers"]) else float(team1["towers"])
        team2_towers = 0.0 if pd.isna(team2["towers"]) else float(team2["towers"])
        team1_barons = 0.0 if pd.isna(team1["barons"]) else float(team1["barons"])
        team2_barons = 0.0 if pd.isna(team2["barons"]) else float(team2["barons"])
        team1_inhibitors = 0.0 if pd.isna(team1["inhibitors"]) else float(team1["inhibitors"])
        team2_inhibitors = 0.0 if pd.isna(team2["inhibitors"]) else float(team2["inhibitors"])
        records.append(
            {
                "snapshot_id": snapshot_dir.name,
                "game_id": str(game_id),
                "event_time": pd.Timestamp(team1["event_time"]),
                "league_code": str(team1["league_code"]),
                "split_name": str(team1["split_name"]) if pd.notna(team1["split_name"]) else str(team1["league_code"]),
                "patch_version": str(team1["patch_version"]) if pd.notna(team1["patch_version"]) else "",
                "playoffs": bool(team1["playoffs"]),
                "team1_key": str(team1["team_key"]),
                "team1_name": str(team1["team_name"]),
                "team2_key": str(team2["team_key"]),
                "team2_name": str(team2["team_name"]),
                "game_length_seconds": float(team1["game_length_seconds"]) if pd.notna(team1["game_length_seconds"]) else 0.0,
                "target_total_kills": float(team1_kills + team2_kills),
                "target_total_dragons": float(team1_dragons + team2_dragons),
                "target_total_towers": float(team1_towers + team2_towers),
                "target_total_barons": float(team1_barons + team2_barons),
                "target_total_inhibitors": float(team1_inhibitors + team2_inhibitors),
                "team1_team_kills": team1_kills,
                "team2_team_kills": team2_kills,
                "team1_dragons": team1_dragons,
                "team2_dragons": team2_dragons,
                "team1_towers": team1_towers,
                "team2_towers": team2_towers,
                "team1_barons": team1_barons,
                "team2_barons": team2_barons,
                "team1_inhibitors": team1_inhibitors,
                "team2_inhibitors": team2_inhibitors,
            }
        )

    pairs_df = pd.DataFrame(records)
    pairs_df["event_date"] = pairs_df["event_time"].dt.floor("D")
    return pairs_df.sort_values(["event_time", "game_id"], kind="stable").reset_index(drop=True)


def build_game_totals_pairs(snapshot_dir: Path) -> pd.DataFrame:
    """Public wrapper for the normalized game-level totals base table."""
    return _build_game_pairs(snapshot_dir)


def build_game_totals_feature_frame(snapshot_dir: Path) -> pd.DataFrame:
    """Build leakage-safe game-level prematch features for totals markets."""
    games_df = _build_game_pairs(snapshot_dir)

    team_history: defaultdict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=max(RECENT_WINDOWS)))
    patch_history: defaultdict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=BASELINE_MAXLEN))
    league_history: defaultdict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=BASELINE_MAXLEN))
    global_history: deque[dict[str, Any]] = deque(maxlen=BASELINE_MAXLEN)

    feature_rows: list[dict[str, Any]] = []

    def team_summary(team_key: str, event_time: pd.Timestamp) -> dict[str, float]:
        history = list(team_history[team_key])
        summary: dict[str, float] = {
            "prior_game_count": float(len(history)),
            "days_since_last_game": _days_since(history[-1]["event_time"], event_time) if history else float("nan"),
        }
        for window in RECENT_WINDOWS:
            recent = history[-window:]
            for metric in FEATURE_METRICS:
                values = [float(item[metric]) for item in recent]
                summary[f"recent{window}_{metric}_mean"] = _weighted_mean(values)
                summary[f"recent{window}_{metric}_var"] = _weighted_var(values)
        return summary

    def baseline_summary(event_time: pd.Timestamp, league_code: str, patch_version: str) -> dict[str, float]:
        patch_items = list(patch_history[patch_version])
        league_items = list(league_history[league_code])
        if len(patch_items) >= BASELINE_MIN_PATCH_SAMPLES:
            chosen = patch_items
            baseline_source = "patch"
        elif len(league_items) >= BASELINE_MIN_LEAGUE_SAMPLES:
            chosen = league_items
            baseline_source = "league"
        else:
            chosen = list(global_history)
            baseline_source = "global"

        summary: dict[str, float] = {
            "baseline_sample_count": float(len(chosen)),
            "baseline_source_patch": float(baseline_source == "patch"),
            "baseline_source_league": float(baseline_source == "league"),
            "baseline_source_global": float(baseline_source == "global"),
        }
        for metric in FEATURE_METRICS:
            values = [float(item[metric]) for item in chosen]
            summary[f"baseline_{metric}_mean"] = _weighted_mean(values)
            summary[f"baseline_{metric}_var"] = _weighted_var(values)
        return summary

    for row in games_df.itertuples(index=False):
        team1_stats = team_summary(row.team1_key, row.event_time)
        team2_stats = team_summary(row.team2_key, row.event_time)
        baseline_stats = baseline_summary(row.event_time, row.league_code, row.patch_version)

        feature_row: dict[str, Any] = {
            "snapshot_id": row.snapshot_id,
            "game_id": row.game_id,
            "event_time": row.event_time,
            "event_date": row.event_date,
            "league_code": row.league_code,
            "split_name": row.split_name,
            "patch_version": row.patch_version,
            "playoffs": row.playoffs,
            "team1_key": row.team1_key,
            "team1_name": row.team1_name,
            "team2_key": row.team2_key,
            "team2_name": row.team2_name,
            **baseline_stats,
        }

        for prefix, stats in (("team1", team1_stats), ("team2", team2_stats)):
            for key, value in stats.items():
                feature_row[f"{prefix}_{key}"] = value

        feature_row["prior_game_count_diff"] = (
            feature_row["team1_prior_game_count"] - feature_row["team2_prior_game_count"]
        )
        feature_row["days_since_last_game_diff"] = (
            feature_row["team1_days_since_last_game"] - feature_row["team2_days_since_last_game"]
        )

        for metric in FEATURE_METRICS:
            for window in RECENT_WINDOWS:
                left_mean = feature_row[f"team1_recent{window}_{metric}_mean"]
                right_mean = feature_row[f"team2_recent{window}_{metric}_mean"]
                left_var = feature_row[f"team1_recent{window}_{metric}_var"]
                right_var = feature_row[f"team2_recent{window}_{metric}_var"]
                feature_row[f"recent{window}_{metric}_mean_avg"] = (left_mean + right_mean) / 2.0
                feature_row[f"recent{window}_{metric}_mean_diff"] = left_mean - right_mean
                feature_row[f"recent{window}_{metric}_var_avg"] = (left_var + right_var) / 2.0

        feature_row["target_total_kills"] = row.target_total_kills
        feature_row["target_total_dragons"] = row.target_total_dragons
        feature_row["target_total_towers"] = row.target_total_towers
        feature_row["target_total_barons"] = row.target_total_barons
        feature_row["target_total_inhibitors"] = row.target_total_inhibitors
        feature_rows.append(feature_row)

        team1_record = {
            "event_time": row.event_time,
            "total_kills": row.target_total_kills,
            "total_dragons": row.target_total_dragons,
            "total_towers": row.target_total_towers,
            "total_barons": row.target_total_barons,
            "total_inhibitors": row.target_total_inhibitors,
            "game_length_seconds": row.game_length_seconds,
            "team_kills": row.team1_team_kills,
            "team_dragons": row.team1_dragons,
            "team_towers": row.team1_towers,
            "team_barons": row.team1_barons,
            "team_inhibitors": row.team1_inhibitors,
        }
        team2_record = {
            "event_time": row.event_time,
            "total_kills": row.target_total_kills,
            "total_dragons": row.target_total_dragons,
            "total_towers": row.target_total_towers,
            "total_barons": row.target_total_barons,
            "total_inhibitors": row.target_total_inhibitors,
            "game_length_seconds": row.game_length_seconds,
            "team_kills": row.team2_team_kills,
            "team_dragons": row.team2_dragons,
            "team_towers": row.team2_towers,
            "team_barons": row.team2_barons,
            "team_inhibitors": row.team2_inhibitors,
        }
        baseline_record = {
            "event_time": row.event_time,
            "total_kills": row.target_total_kills,
            "total_dragons": row.target_total_dragons,
            "total_towers": row.target_total_towers,
            "total_barons": row.target_total_barons,
            "total_inhibitors": row.target_total_inhibitors,
            "game_length_seconds": row.game_length_seconds,
            "team_kills": (row.team1_team_kills + row.team2_team_kills) / 2.0,
            "team_dragons": (row.team1_dragons + row.team2_dragons) / 2.0,
            "team_towers": (row.team1_towers + row.team2_towers) / 2.0,
            "team_barons": (row.team1_barons + row.team2_barons) / 2.0,
            "team_inhibitors": (row.team1_inhibitors + row.team2_inhibitors) / 2.0,
        }

        team_history[row.team1_key].append(team1_record)
        team_history[row.team2_key].append(team2_record)
        patch_history[row.patch_version].append(baseline_record)
        league_history[row.league_code].append(baseline_record)
        global_history.append(baseline_record)

    feature_df = pd.DataFrame(feature_rows)
    feature_df = feature_df.dropna(subset=TARGET_COLUMNS)
    return feature_df.sort_values(["event_time", "game_id"], kind="stable").reset_index(drop=True)


def build_game_totals_team_history_frame(games_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten game-level pairs into team-centric history rows for future scoring."""
    team_frames: list[pd.DataFrame] = []
    for side in (1, 2):
        team_frame = games_df[
            [
                "game_id",
                "event_time",
                "league_code",
                "split_name",
                "patch_version",
                "game_length_seconds",
                "target_total_kills",
                "target_total_dragons",
                "target_total_towers",
                "target_total_barons",
                "target_total_inhibitors",
                f"team{side}_key",
                f"team{side}_name",
                f"team{side}_team_kills",
                f"team{side}_dragons",
                f"team{side}_towers",
                f"team{side}_barons",
                f"team{side}_inhibitors",
            ]
        ].copy()
        team_frame = team_frame.rename(
            columns={
                f"team{side}_key": "team_key",
                f"team{side}_name": "team_name",
                "target_total_kills": "total_kills",
                "target_total_dragons": "total_dragons",
                "target_total_towers": "total_towers",
                "target_total_barons": "total_barons",
                "target_total_inhibitors": "total_inhibitors",
                f"team{side}_team_kills": "team_kills",
                f"team{side}_dragons": "team_dragons",
                f"team{side}_towers": "team_towers",
                f"team{side}_barons": "team_barons",
                f"team{side}_inhibitors": "team_inhibitors",
            }
        )
        team_frames.append(team_frame)

    return (
        pd.concat(team_frames, ignore_index=True)
        .sort_values(["event_time", "game_id", "team_key"], kind="stable")
        .reset_index(drop=True)
    )


def _team_summary_from_history(
    team_history_df: pd.DataFrame,
    team_key: str,
    event_time: pd.Timestamp,
) -> dict[str, float]:
    history = (
        team_history_df.loc[
            (team_history_df["team_key"] == team_key) & (team_history_df["event_time"] < event_time)
        ]
        .sort_values(["event_time", "game_id"], kind="stable")
        .tail(max(RECENT_WINDOWS))
    )
    last_seen = history["event_time"].iloc[-1] if not history.empty else None
    summary: dict[str, float] = {
        "prior_game_count": float(len(history)),
        "days_since_last_game": _days_since(last_seen, event_time),
    }
    for window in RECENT_WINDOWS:
        recent = history.tail(window)
        for metric in FEATURE_METRICS:
            values = recent[metric].astype(float).tolist() if metric in recent.columns else []
            summary[f"recent{window}_{metric}_mean"] = _weighted_mean(values)
            summary[f"recent{window}_{metric}_var"] = _weighted_var(values)
    return summary


def _baseline_summary_from_games(
    games_df: pd.DataFrame,
    event_time: pd.Timestamp,
    league_code: str,
    patch_version: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    prior_games = games_df.loc[games_df["event_time"] < event_time].sort_values(
        ["event_time", "game_id"], kind="stable"
    )
    patch_items = prior_games.loc[prior_games["patch_version"] == patch_version].tail(BASELINE_MAXLEN)
    league_items = prior_games.loc[prior_games["league_code"] == league_code].tail(BASELINE_MAXLEN)
    global_items = prior_games.tail(BASELINE_MAXLEN)

    if len(patch_items) >= BASELINE_MIN_PATCH_SAMPLES:
        chosen = patch_items
        baseline_source = "patch"
    elif len(league_items) >= BASELINE_MIN_LEAGUE_SAMPLES:
        chosen = league_items
        baseline_source = "league"
    else:
        chosen = global_items
        baseline_source = "global"

    summary: dict[str, float] = {
        "baseline_sample_count": float(len(chosen)),
        "baseline_source_patch": float(baseline_source == "patch"),
        "baseline_source_league": float(baseline_source == "league"),
        "baseline_source_global": float(baseline_source == "global"),
    }
    for metric in FEATURE_METRICS:
        values = chosen[metric].astype(float).tolist() if metric in chosen.columns else []
        summary[f"baseline_{metric}_mean"] = _weighted_mean(values)
        summary[f"baseline_{metric}_var"] = _weighted_var(values)

    return summary, {
        "baseline_source": baseline_source,
        "patch_sample_count": int(len(patch_items)),
        "league_sample_count": int(len(league_items)),
        "global_sample_count": int(len(global_items)),
    }


def build_upcoming_game_totals_feature_row(
    games_df: pd.DataFrame,
    team_history_df: pd.DataFrame,
    *,
    snapshot_id: str,
    team1_key: str,
    team1_name: str,
    team2_key: str,
    team2_name: str,
    event_time: pd.Timestamp,
    league_code: str,
    split_name: str,
    patch_version: str,
    playoffs: bool = False,
    game_id: str = "upcoming",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one prematch feature row with the same schema used by totals training."""
    event_time = pd.Timestamp(event_time)
    if event_time.tzinfo is None:
        event_time = event_time.tz_localize("UTC")
    else:
        event_time = event_time.tz_convert("UTC")

    baseline_stats, baseline_meta = _baseline_summary_from_games(
        games_df,
        event_time=event_time,
        league_code=league_code,
        patch_version=patch_version,
    )
    team1_stats = _team_summary_from_history(team_history_df, team1_key, event_time)
    team2_stats = _team_summary_from_history(team_history_df, team2_key, event_time)

    feature_row: dict[str, Any] = {
        "snapshot_id": snapshot_id,
        "game_id": game_id,
        "event_time": event_time,
        "event_date": event_time.floor("D"),
        "league_code": league_code,
        "split_name": split_name or league_code,
        "patch_version": patch_version or "",
        "playoffs": bool(playoffs),
        "team1_key": team1_key,
        "team1_name": team1_name,
        "team2_key": team2_key,
        "team2_name": team2_name,
        **baseline_stats,
    }
    for prefix, stats in (("team1", team1_stats), ("team2", team2_stats)):
        for key, value in stats.items():
            feature_row[f"{prefix}_{key}"] = value

    feature_row["prior_game_count_diff"] = feature_row["team1_prior_game_count"] - feature_row["team2_prior_game_count"]
    feature_row["days_since_last_game_diff"] = (
        feature_row["team1_days_since_last_game"] - feature_row["team2_days_since_last_game"]
    )

    for metric in FEATURE_METRICS:
        for window in RECENT_WINDOWS:
            left_mean = feature_row[f"team1_recent{window}_{metric}_mean"]
            right_mean = feature_row[f"team2_recent{window}_{metric}_mean"]
            left_var = feature_row[f"team1_recent{window}_{metric}_var"]
            right_var = feature_row[f"team2_recent{window}_{metric}_var"]
            feature_row[f"recent{window}_{metric}_mean_avg"] = (left_mean + right_mean) / 2.0
            feature_row[f"recent{window}_{metric}_mean_diff"] = left_mean - right_mean
            feature_row[f"recent{window}_{metric}_var_avg"] = (left_var + right_var) / 2.0

    metadata = {
        **baseline_meta,
        "baseline_sample_count": int(baseline_stats["baseline_sample_count"]),
        "team1_prior_game_count": int(team1_stats["prior_game_count"]),
        "team2_prior_game_count": int(team2_stats["prior_game_count"]),
    }
    return pd.DataFrame([feature_row]), metadata


def _build_preprocessor(feature_df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_columns = [column for column in CATEGORICAL_COLUMNS if column in feature_df.columns]
    excluded = set(META_COLUMNS + TARGET_COLUMNS)
    numeric_columns = [
        column for column in feature_df.columns if column not in excluded and column not in categorical_columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
        ],
        remainder="drop",
    )
    return preprocessor, categorical_columns, numeric_columns


def _build_model_registry(random_state: int = 42) -> dict[str, Any]:
    registry: dict[str, Any] = {
        "ridge": Ridge(alpha=1.0),
    }
    if XGBRegressor is not None:
        registry["xgboost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            tree_method="hist",
        )
    else:
        logger.warning("xgboost not available; training only ridge baseline for totals")
    return registry


def _build_pipeline(estimator: Any, preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def _extract_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in META_COLUMNS + TARGET_COLUMNS]


def _compute_regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "target_mean": float(pd.Series(y_true).mean()),
    }


def _estimate_overdispersion_alpha(y_true: pd.Series, predictions: np.ndarray) -> float:
    mu = np.clip(np.asarray(predictions, dtype=float), 1e-6, None)
    y = np.asarray(y_true, dtype=float)
    numerator = float(np.mean(np.maximum((y - mu) ** 2 - mu, 0.0)))
    denominator = float(np.mean(mu**2))
    if denominator <= 0.0:
        return 0.0
    return max(0.0, numerator / denominator)


def run_game_totals_baseline(
    snapshot_dir: Path | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    train_end_date: str = DEFAULT_TRAIN_END_DATE,
    validation_start_date: str = DEFAULT_VALIDATION_START_DATE,
    validation_end_date: str = DEFAULT_VALIDATION_END_DATE,
    test_start_date: str = DEFAULT_TEST_START_DATE,
    random_state: int = 42,
) -> GameTotalsRunResult:
    """Train regression baselines for totals markets on the latest Gold snapshot."""
    if snapshot_dir is None:
        snapshot_id, resolved_snapshot_dir = resolve_latest_snapshot()
    else:
        resolved_snapshot_dir = Path(snapshot_dir)
        snapshot_id = resolved_snapshot_dir.name

    run_id = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_df = build_game_totals_feature_frame(resolved_snapshot_dir)
    features_path = output_dir / "game_totals_features.parquet"
    feature_df.to_parquet(features_path, index=False)

    train_df, validation_df, test_df = create_calendar_train_validation_test_split(
        feature_df,
        train_end_date=train_end_date,
        validation_start_date=validation_start_date,
        validation_end_date=validation_end_date,
        test_start_date=test_start_date,
    )
    final_train_df = pd.concat([train_df, validation_df], ignore_index=True)

    feature_columns = _extract_feature_columns(feature_df)
    base_preprocessor, _, _ = _build_preprocessor(feature_df[feature_columns])
    models = _build_model_registry(random_state=random_state)

    metrics_records: list[dict[str, Any]] = []
    train_prediction_records: list[pd.DataFrame] = []
    test_prediction_records: list[pd.DataFrame] = []
    best_models: dict[str, Pipeline] = {}
    dispersion_params: dict[str, float] = {}

    meta_columns = ["game_id", "event_time", "event_date", "league_code", "team1_name", "team2_name"]

    for target_column in TARGET_COLUMNS:
        validation_candidates: list[dict[str, Any]] = []
        candidate_models: dict[str, Pipeline] = {}

        for model_name, estimator in models.items():
            pipeline = _build_pipeline(clone(estimator), clone(base_preprocessor))
            pipeline.fit(train_df[feature_columns], train_df[target_column])
            validation_predictions = pipeline.predict(validation_df[feature_columns])
            metrics = _compute_regression_metrics(validation_df[target_column], validation_predictions)
            validation_candidates.append(
                {
                    "target": target_column,
                    "market": TARGET_TO_MARKET[target_column],
                    "stage": "validation",
                    "model_name": model_name,
                    "rows": int(len(validation_df)),
                    **metrics,
                }
            )
            candidate_models[model_name] = pipeline

        validation_df_metrics = pd.DataFrame(validation_candidates).sort_values(["mae", "rmse"])
        best_model_name = str(validation_df_metrics.iloc[0]["model_name"])

        final_pipeline = _build_pipeline(clone(models[best_model_name]), clone(base_preprocessor))
        final_pipeline.fit(final_train_df[feature_columns], final_train_df[target_column])
        final_train_predictions = final_pipeline.predict(final_train_df[feature_columns])
        test_predictions = final_pipeline.predict(test_df[feature_columns])
        dispersion_alpha = _estimate_overdispersion_alpha(final_train_df[target_column], final_train_predictions)

        train_pred_frame = final_train_df[meta_columns].copy()
        train_pred_frame["target"] = target_column
        train_pred_frame["market"] = TARGET_TO_MARKET[target_column]
        train_pred_frame["model_name"] = best_model_name
        train_pred_frame["actual_value"] = final_train_df[target_column].to_numpy()
        train_pred_frame["predicted_value"] = final_train_predictions
        train_prediction_records.append(train_pred_frame)

        test_pred_frame = test_df[meta_columns].copy()
        test_pred_frame["target"] = target_column
        test_pred_frame["market"] = TARGET_TO_MARKET[target_column]
        test_pred_frame["model_name"] = best_model_name
        test_pred_frame["actual_value"] = test_df[target_column].to_numpy()
        test_pred_frame["predicted_value"] = test_predictions
        test_prediction_records.append(test_pred_frame)

        validation_metrics = _compute_regression_metrics(validation_df[target_column], candidate_models[best_model_name].predict(validation_df[feature_columns]))
        test_metrics = _compute_regression_metrics(test_df[target_column], test_predictions)
        metrics_records.extend(
            [
                {
                    "target": target_column,
                    "market": TARGET_TO_MARKET[target_column],
                    "stage": "validation",
                    "model_name": best_model_name,
                    "rows": int(len(validation_df)),
                    "dispersion_alpha": float(dispersion_alpha),
                    **validation_metrics,
                },
                {
                    "target": target_column,
                    "market": TARGET_TO_MARKET[target_column],
                    "stage": "test",
                    "model_name": best_model_name,
                    "rows": int(len(test_df)),
                    "dispersion_alpha": float(dispersion_alpha),
                    **test_metrics,
                },
            ]
        )

        model_path = output_dir / f"{target_column}_{best_model_name}.pkl"
        with model_path.open("wb") as file_obj:
            pickle.dump(final_pipeline, file_obj)

        best_models[target_column] = final_pipeline
        dispersion_params[target_column] = float(dispersion_alpha)

    metrics_df = pd.DataFrame(metrics_records).sort_values(["target", "stage"])
    metrics_path = output_dir / "metrics.json"
    metrics_payload = {
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "features_path": str(features_path),
        "split": {
            "train_end_date": train_end_date,
            "validation_start_date": validation_start_date,
            "validation_end_date": validation_end_date,
            "test_start_date": test_start_date,
            "rows_total": int(len(feature_df)),
            "rows_train": int(len(train_df)),
            "rows_validation": int(len(validation_df)),
            "rows_test": int(len(test_df)),
        },
        "targets": {
            target: {
                "market": TARGET_TO_MARKET[target],
                "best_model_name": str(
                    metrics_df.loc[(metrics_df["target"] == target) & (metrics_df["stage"] == "test"), "model_name"].iloc[0]
                ),
                "dispersion_alpha": float(dispersion_params[target]),
            }
            for target in TARGET_COLUMNS
        },
        "metrics": metrics_df.to_dict(orient="records"),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    train_predictions_path = output_dir / "train_predictions.parquet"
    pd.concat(train_prediction_records, ignore_index=True).to_parquet(train_predictions_path, index=False)

    test_predictions_path = output_dir / "test_predictions.parquet"
    pd.concat(test_prediction_records, ignore_index=True).to_parquet(test_predictions_path, index=False)

    best_models_path = output_dir / "best_models.json"
    best_models_payload = {
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "targets": {
            target: {
                "market": TARGET_TO_MARKET[target],
                "model_path": str(output_dir / f"{target}_{metrics_payload['targets'][target]['best_model_name']}.pkl"),
                "model_name": metrics_payload["targets"][target]["best_model_name"],
                "dispersion_alpha": metrics_payload["targets"][target]["dispersion_alpha"],
            }
            for target in TARGET_COLUMNS
        },
    }
    best_models_path.write_text(json.dumps(best_models_payload, indent=2), encoding="utf-8")

    latest_run_pointer = output_root / "latest_run.json"
    latest_run_pointer.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "snapshot_id": snapshot_id,
                "metrics_path": str(metrics_path),
                "best_models_path": str(best_models_path),
                "generated_at": _utc_now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return GameTotalsRunResult(
        run_id=run_id,
        output_dir=output_dir,
        snapshot_id=snapshot_id,
        features_path=features_path,
        metrics_path=metrics_path,
        train_predictions_path=train_predictions_path,
        test_predictions_path=test_predictions_path,
        best_models_path=best_models_path,
    )
