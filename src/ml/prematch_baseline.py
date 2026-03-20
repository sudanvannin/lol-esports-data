"""Temporal baseline training pipeline for Gold prematch features."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - environment dependent
    XGBClassifier = None


logger = logging.getLogger(__name__)

LATEST_SNAPSHOT_POINTER = Path("data/gold/latest_snapshot.json")
DEFAULT_OUTPUT_ROOT = Path("data/models/prematch_baseline")
DEFAULT_SPLIT_STRATEGY = "calendar"
DEFAULT_TRAIN_END_DATE = "2023-12-31"
DEFAULT_VALIDATION_START_DATE = "2024-01-01"
DEFAULT_VALIDATION_END_DATE = "2024-12-31"
DEFAULT_TEST_START_DATE = "2025-01-01"
LABEL_COLUMN = "label_team1_win"
META_COLUMNS = [
    "snapshot_id",
    "feature_version",
    "series_key",
    "series_date",
    "start_time",
    "team1_name",
    "team2_name",
]
CATEGORICAL_COLUMNS = [
    "league_code",
    "split_name",
    "patch_version",
    "team1_key",
    "team2_key",
]


@dataclass(slots=True)
class TemporalFold:
    """Chronological train/test split over event dates."""

    fold_id: int
    train_dates: list[str]
    test_dates: list[str]


@dataclass(slots=True)
class BaselineRunResult:
    """Artifacts produced by the baseline training pipeline."""

    run_id: str
    output_dir: Path
    snapshot_id: str
    features_path: Path
    metrics_path: Path
    selection_metrics_path: Path
    fold_metrics_path: Path
    holdout_predictions_path: Path
    league_metrics_path: Path
    calibration_bins_path: Path
    best_model_path: Path
    best_model_name: str


def _utc_now() -> datetime:
    return datetime.now(UTC)


def resolve_latest_features_path(pointer_path: Path = LATEST_SNAPSHOT_POINTER) -> tuple[str, Path]:
    """Resolve the most recent Gold prematch feature table."""
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    snapshot_id = pointer["snapshot_id"]
    snapshot_dir = Path(pointer["snapshot_dir"])
    features_path = snapshot_dir / "match_features_prematch.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Prematch features not found at {features_path}")
    return snapshot_id, features_path


def load_feature_frame(features_path: Path) -> pd.DataFrame:
    """Load prematch features and normalize event timestamps."""
    df = pd.read_parquet(features_path)
    df["series_date"] = pd.to_datetime(df["series_date"], utc=True, errors="coerce")
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df["event_time"] = df["start_time"].fillna(df["series_date"])
    df["event_date"] = df["event_time"].dt.floor("D")
    df = df.sort_values(["event_time", "series_key"], kind="stable").reset_index(drop=True)
    return df


def _build_temporal_folds(event_dates: pd.Series, n_splits: int) -> list[TemporalFold]:
    """Create expanding-window folds grouped by event date."""
    unique_dates = pd.Index(sorted(pd.to_datetime(event_dates.dropna().unique(), utc=True)))
    if unique_dates.empty:
        raise ValueError("No valid event dates available for temporal split")

    max_splits = max(1, len(unique_dates) - 1)
    effective_splits = min(n_splits, max_splits)
    date_segments = [segment for segment in np.array_split(unique_dates, effective_splits + 1) if len(segment)]

    folds: list[TemporalFold] = []
    for fold_id in range(len(date_segments) - 1):
        train_dates = pd.Index(np.concatenate(date_segments[: fold_id + 1]))
        test_dates = pd.Index(date_segments[fold_id + 1])
        if train_dates.empty or test_dates.empty:
            continue
        folds.append(
            TemporalFold(
                fold_id=fold_id + 1,
                train_dates=[ts.isoformat() for ts in train_dates],
                test_dates=[ts.isoformat() for ts in test_dates],
            )
        )
    return folds


def create_temporal_train_holdout_split(
    df: pd.DataFrame,
    holdout_fraction: float = 0.2,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, list[TemporalFold]]:
    """Split data into development and holdout sets using chronological event dates."""
    if not 0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5")

    unique_dates = pd.Index(sorted(pd.to_datetime(df["event_date"].dropna().unique(), utc=True)))
    if len(unique_dates) < 3:
        raise ValueError("Need at least 3 unique event dates for temporal evaluation")

    holdout_dates_count = max(1, int(np.ceil(len(unique_dates) * holdout_fraction)))
    dev_dates = unique_dates[:-holdout_dates_count]
    holdout_dates = unique_dates[-holdout_dates_count:]
    if len(dev_dates) < 2:
        raise ValueError("Holdout split left too little development history")

    dev_mask = df["event_date"].isin(dev_dates)
    holdout_mask = df["event_date"].isin(holdout_dates)
    dev_df = df.loc[dev_mask].copy()
    holdout_df = df.loc[holdout_mask].copy()
    folds = _build_temporal_folds(dev_df["event_date"], n_splits=n_splits)
    return dev_df, holdout_df, folds


def create_calendar_train_validation_test_split(
    df: pd.DataFrame,
    train_end_date: str = DEFAULT_TRAIN_END_DATE,
    validation_start_date: str = DEFAULT_VALIDATION_START_DATE,
    validation_end_date: str = DEFAULT_VALIDATION_END_DATE,
    test_start_date: str = DEFAULT_TEST_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into explicit calendar-based train, validation, and test windows."""
    train_end = pd.Timestamp(train_end_date, tz="UTC").floor("D")
    validation_start = pd.Timestamp(validation_start_date, tz="UTC").floor("D")
    validation_end = pd.Timestamp(validation_end_date, tz="UTC").floor("D")
    test_start = pd.Timestamp(test_start_date, tz="UTC").floor("D")

    if not train_end < validation_start:
        raise ValueError("train_end_date must be earlier than validation_start_date")
    if not validation_start <= validation_end:
        raise ValueError("validation_start_date must be earlier than validation_end_date")
    if not validation_end < test_start:
        raise ValueError("validation_end_date must be earlier than test_start_date")

    train_df = df.loc[df["event_date"] <= train_end].copy()
    validation_df = df.loc[
        (df["event_date"] >= validation_start) & (df["event_date"] <= validation_end)
    ].copy()
    test_df = df.loc[df["event_date"] >= test_start].copy()

    if train_df.empty:
        raise ValueError("Calendar split produced an empty training window")
    if validation_df.empty:
        raise ValueError("Calendar split produced an empty validation window")
    if test_df.empty:
        raise ValueError("Calendar split produced an empty test window")

    return train_df, validation_df, test_df


def _build_preprocessor(feature_df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_columns = [col for col in CATEGORICAL_COLUMNS if col in feature_df.columns]
    excluded = set(META_COLUMNS + [LABEL_COLUMN, "event_time", "event_date"])
    numeric_columns = [
        col
        for col in feature_df.columns
        if col not in excluded and col not in categorical_columns
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
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            solver="saga",
            C=1.0,
            random_state=random_state,
        )
    }
    if XGBClassifier is not None:
        registry["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            tree_method="hist",
        )
    else:
        logger.warning("xgboost not available; training only logistic_regression baseline")
    return registry


def _build_pipeline(estimator: Any, preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def _extract_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in META_COLUMNS + [LABEL_COLUMN, "event_time", "event_date"]]


def _compute_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    clipped_probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    roc_auc = float("nan")
    if pd.Series(y_true).nunique() >= 2:
        roc_auc = float(roc_auc_score(y_true, clipped_probabilities))
    return {
        "log_loss": float(log_loss(y_true, clipped_probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, clipped_probabilities)),
        "roc_auc": roc_auc,
    }


def _evaluate_temporal_cv(
    dev_df: pd.DataFrame,
    folds: list[TemporalFold],
    models: dict[str, Any],
) -> pd.DataFrame:
    feature_columns = _extract_feature_columns(dev_df)
    fold_records: list[dict[str, Any]] = []

    for model_name, estimator in models.items():
        for fold in folds:
            train_mask = dev_df["event_date"].isin(pd.to_datetime(fold.train_dates, utc=True))
            test_mask = dev_df["event_date"].isin(pd.to_datetime(fold.test_dates, utc=True))
            train_df = dev_df.loc[train_mask]
            test_df = dev_df.loc[test_mask]

            preprocessor, _, _ = _build_preprocessor(train_df[feature_columns])
            pipeline = _build_pipeline(estimator, preprocessor)
            pipeline.fit(train_df[feature_columns], train_df[LABEL_COLUMN])
            probabilities = pipeline.predict_proba(test_df[feature_columns])[:, 1]
            metrics = _compute_metrics(test_df[LABEL_COLUMN], probabilities)

            fold_records.append(
                {
                    "model_name": model_name,
                    "evaluation_stage": "temporal_cv",
                    "fold_id": fold.fold_id,
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "train_start_date": str(train_df["event_date"].min().date()),
                    "train_end_date": str(train_df["event_date"].max().date()),
                    "test_start_date": str(test_df["event_date"].min().date()),
                    "test_end_date": str(test_df["event_date"].max().date()),
                    **metrics,
                }
            )

    return pd.DataFrame(fold_records)


def _evaluate_model_window(
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    models: dict[str, Any],
    evaluation_stage: str,
) -> pd.DataFrame:
    feature_columns = _extract_feature_columns(train_df)
    records: list[dict[str, Any]] = []

    for model_name, estimator in models.items():
        preprocessor, _, _ = _build_preprocessor(train_df[feature_columns])
        pipeline = _build_pipeline(estimator, preprocessor)
        pipeline.fit(train_df[feature_columns], train_df[LABEL_COLUMN])
        probabilities = pipeline.predict_proba(evaluation_df[feature_columns])[:, 1]
        metrics = _compute_metrics(evaluation_df[LABEL_COLUMN], probabilities)

        records.append(
            {
                "model_name": model_name,
                "evaluation_stage": evaluation_stage,
                "fold_id": 1,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(evaluation_df)),
                "train_start_date": str(train_df["event_date"].min().date()),
                "train_end_date": str(train_df["event_date"].max().date()),
                "test_start_date": str(evaluation_df["event_date"].min().date()),
                "test_end_date": str(evaluation_df["event_date"].max().date()),
                **metrics,
            }
        )

    return pd.DataFrame(records)


def _summarize_selection_metrics(raw_selection_df: pd.DataFrame) -> pd.DataFrame:
    if raw_selection_df.empty:
        raise ValueError("No selection metrics available")

    summary_df = (
        raw_selection_df.groupby("model_name", as_index=False)
        .agg(
            evaluation_stage=("evaluation_stage", "first"),
            evaluation_runs=("fold_id", "count"),
            evaluation_rows=("test_rows", "sum"),
            selection_start_date=("test_start_date", "min"),
            selection_end_date=("test_end_date", "max"),
            log_loss=("log_loss", "mean"),
            brier_score=("brier_score", "mean"),
            roc_auc=("roc_auc", "mean"),
        )
        .sort_values(["log_loss", "brier_score", "roc_auc"], na_position="last")
        .reset_index(drop=True)
    )
    return summary_df


def _fit_final_models(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    models: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    feature_columns = _extract_feature_columns(train_df)
    model_predictions: list[pd.DataFrame] = []
    fitted_models: dict[str, Pipeline] = {}

    holdout_meta = holdout_df[
        [
            "series_key",
            "series_date",
            "start_time",
            "league_code",
            "team1_key",
            "team2_key",
            LABEL_COLUMN,
        ]
    ].copy()

    for model_name, estimator in models.items():
        preprocessor, _, _ = _build_preprocessor(train_df[feature_columns])
        pipeline = _build_pipeline(estimator, preprocessor)
        pipeline.fit(train_df[feature_columns], train_df[LABEL_COLUMN])
        probabilities = pipeline.predict_proba(holdout_df[feature_columns])[:, 1]
        model_frame = holdout_meta.copy()
        model_frame["model_name"] = model_name
        model_frame["pred_team1_win_prob"] = probabilities
        model_predictions.append(model_frame)
        fitted_models[model_name] = pipeline

    predictions_df = pd.concat(model_predictions, ignore_index=True)
    return predictions_df, fitted_models


def _summarize_holdout_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model_name, frame in predictions_df.groupby("model_name", sort=False):
        metrics = _compute_metrics(frame[LABEL_COLUMN], frame["pred_team1_win_prob"])
        records.append(
            {
                "model_name": model_name,
                "rows": int(len(frame)),
                "test_start_date": str(pd.to_datetime(frame["series_date"]).min().date()),
                "test_end_date": str(pd.to_datetime(frame["series_date"]).max().date()),
                **metrics,
            }
        )
    return pd.DataFrame(records).sort_values(["log_loss", "brier_score", "roc_auc"], na_position="last")


def _build_league_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (model_name, league_code), frame in predictions_df.groupby(["model_name", "league_code"], sort=False):
        if frame[LABEL_COLUMN].nunique() < 2:
            continue
        metrics = _compute_metrics(frame[LABEL_COLUMN], frame["pred_team1_win_prob"])
        records.append(
            {
                "model_name": model_name,
                "league_code": league_code,
                "rows": int(len(frame)),
                **metrics,
            }
        )
    return pd.DataFrame(records).sort_values(["model_name", "league_code"])


def _build_calibration_bins(predictions_df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    records: list[dict[str, Any]] = []
    for model_name, frame in predictions_df.groupby("model_name", sort=False):
        working = frame.copy()
        working["probability_bin"] = pd.cut(
            working["pred_team1_win_prob"],
            bins=bin_edges,
            include_lowest=True,
            duplicates="drop",
        )
        for bucket, bucket_frame in working.groupby("probability_bin", observed=False):
            if bucket_frame.empty:
                continue
            records.append(
                {
                    "model_name": model_name,
                    "probability_bin": str(bucket),
                    "rows": int(len(bucket_frame)),
                    "avg_pred_team1_win_prob": float(bucket_frame["pred_team1_win_prob"].mean()),
                    "actual_team1_win_rate": float(bucket_frame[LABEL_COLUMN].mean()),
                }
            )
    return pd.DataFrame(records)


def _pick_best_model(selection_metrics_df: pd.DataFrame) -> str:
    if selection_metrics_df.empty:
        raise ValueError("No selection metrics available to choose a model")
    return str(
        selection_metrics_df.sort_values(["log_loss", "brier_score"], na_position="last").iloc[0][
            "model_name"
        ]
    )


def run_prematch_baseline(
    features_path: Path | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    holdout_fraction: float = 0.2,
    n_splits: int = 5,
    train_end_date: str = DEFAULT_TRAIN_END_DATE,
    validation_start_date: str = DEFAULT_VALIDATION_START_DATE,
    validation_end_date: str = DEFAULT_VALIDATION_END_DATE,
    test_start_date: str = DEFAULT_TEST_START_DATE,
    random_state: int = 42,
) -> BaselineRunResult:
    """Train and evaluate baseline models on the Gold prematch feature table."""
    if features_path is None:
        snapshot_id, resolved_features_path = resolve_latest_features_path()
    else:
        resolved_features_path = Path(features_path)
        snapshot_id = resolved_features_path.parent.name

    if not resolved_features_path.exists():
        raise FileNotFoundError(f"Feature table not found at {resolved_features_path}")

    run_id = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_frame(resolved_features_path)
    models = _build_model_registry(random_state=random_state)

    if split_strategy == "calendar":
        train_df, validation_df, holdout_df = create_calendar_train_validation_test_split(
            df,
            train_end_date=train_end_date,
            validation_start_date=validation_start_date,
            validation_end_date=validation_end_date,
            test_start_date=test_start_date,
        )
        fold_metrics_df = _evaluate_model_window(
            train_df,
            validation_df,
            models,
            evaluation_stage="validation_window",
        )
        selection_metrics_df = _summarize_selection_metrics(fold_metrics_df)
        final_train_df = pd.concat([train_df, validation_df], ignore_index=True)
        split_metadata: dict[str, Any] = {
            "split_strategy": split_strategy,
            "train_end_date": train_end_date,
            "validation_start_date": validation_start_date,
            "validation_end_date": validation_end_date,
            "test_start_date": test_start_date,
            "rows_train": int(len(train_df)),
            "rows_validation": int(len(validation_df)),
            "rows_test": int(len(holdout_df)),
        }
    elif split_strategy == "fraction":
        dev_df, holdout_df, folds = create_temporal_train_holdout_split(
            df,
            holdout_fraction=holdout_fraction,
            n_splits=n_splits,
        )
        fold_metrics_df = _evaluate_temporal_cv(dev_df, folds, models)
        selection_metrics_df = _summarize_selection_metrics(fold_metrics_df)
        final_train_df = dev_df
        split_metadata = {
            "split_strategy": split_strategy,
            "holdout_fraction": holdout_fraction,
            "n_splits": n_splits,
            "rows_development": int(len(dev_df)),
            "rows_test": int(len(holdout_df)),
            "folds": [asdict(fold) for fold in folds],
        }
    else:
        raise ValueError("split_strategy must be either 'calendar' or 'fraction'")

    predictions_df, fitted_models = _fit_final_models(final_train_df, holdout_df, models)
    holdout_metrics_df = _summarize_holdout_metrics(predictions_df)
    league_metrics_df = _build_league_metrics(predictions_df)
    calibration_bins_df = _build_calibration_bins(predictions_df)
    best_model_name = _pick_best_model(selection_metrics_df)

    best_model_path = output_dir / f"{best_model_name}.pkl"
    with best_model_path.open("wb") as file_obj:
        pickle.dump(fitted_models[best_model_name], file_obj)

    metrics_payload = {
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "features_path": str(resolved_features_path),
        "rows_total": int(len(df)),
        "models": list(models.keys()),
        "split": split_metadata,
        "best_model_name": best_model_name,
        "selection_metrics": selection_metrics_df.to_dict(orient="records"),
        "test_metrics": holdout_metrics_df.to_dict(orient="records"),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    selection_metrics_path = output_dir / "selection_metrics.parquet"
    fold_metrics_path = output_dir / "fold_metrics.parquet"
    holdout_predictions_path = output_dir / "holdout_predictions.parquet"
    league_metrics_path = output_dir / "league_metrics.parquet"
    calibration_bins_path = output_dir / "calibration_bins.parquet"

    selection_metrics_df.to_parquet(selection_metrics_path, index=False)
    fold_metrics_df.to_parquet(fold_metrics_path, index=False)
    predictions_df.to_parquet(holdout_predictions_path, index=False)
    league_metrics_df.to_parquet(league_metrics_path, index=False)
    calibration_bins_df.to_parquet(calibration_bins_path, index=False)

    latest_pointer_path = output_root / "latest_run.json"
    latest_pointer = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "features_path": str(resolved_features_path),
        "snapshot_id": snapshot_id,
        "generated_at": _utc_now().isoformat(),
    }
    latest_pointer_path.write_text(json.dumps(latest_pointer, indent=2), encoding="utf-8")

    return BaselineRunResult(
        run_id=run_id,
        output_dir=output_dir,
        snapshot_id=snapshot_id,
        features_path=resolved_features_path,
        metrics_path=metrics_path,
        selection_metrics_path=selection_metrics_path,
        fold_metrics_path=fold_metrics_path,
        holdout_predictions_path=holdout_predictions_path,
        league_metrics_path=league_metrics_path,
        calibration_bins_path=calibration_bins_path,
        best_model_path=best_model_path,
        best_model_name=best_model_name,
    )
