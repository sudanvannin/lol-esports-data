# Prematch Training

## Objective

This project trains a first baseline model to predict whether `team1` wins a series using only information available before the series starts.

The current baseline is designed to answer one question first:

> Do the Gold prematch features contain real predictive signal?

This phase is not the final production model. It is a reproducible benchmark used to validate data quality, feature usefulness, and temporal evaluation logic.

## Training Dataset

The training dataset is generated from the Gold snapshot pipeline and exported as:

- `data/gold/snapshots/<snapshot_id>/match_features_prematch.parquet`

As of snapshot `20260316T003217Z`, the active feature table is:

- `data/gold/snapshots/20260316T003217Z/match_features_prematch.parquet`

Current scope:

- Leagues: `LCK`, `LEC`, `LPL`, `LCS`, `CBLOL`, `LTA`, `LTA S`, `LTA N`, `MSI`, `WLDs`, `FST`
- Grain: `1 row = 1 series`
- Label: `label_team1_win`
- Current row count: `7,906`

## Feature Set

The prematch table is built in `src/transform/gold_layer.py` and contains only leakage-safe features.

Main feature groups:

- pre-series Elo and implied win probability
- prior series and game win rates
- recent form over the last 5 series
- average games played and games won per series
- average game length
- streak and days since previous series
- patch-level history
- split-level history
- head-to-head series and game history
- roster continuity and new-player pressure
- average prior experience and win rate of the current roster
- recent draft champion-pool breadth
- recent draft concentration and first-pick tendencies

The feature export is produced inside:

- `src/transform/gold_layer.py`

The training pipeline consumes this table directly and does not rebuild features on the fly.

## Why These Algorithms

The baseline starts with `LogisticRegression` for pragmatic reasons:

- binary target
- direct probability output
- strong baseline for tabular data
- cheap to train in CPU
- easy to debug and calibrate

`XGBoost` is also enabled and is the current best baseline because it captures non-linear interactions between league, roster, patch, form, and draft-history features better than a linear model.

At the current dataset size, GPU is not required. CPU training is sufficient.

## Temporal Validation Strategy

The project does not use random train/test split.

Reason:

- esports data drifts over time
- patch changes alter the meta
- rosters change
- league formats change
- random shuffling leaks future structure into training

Default evaluation logic:

1. Train on all series up to `December 31, 2023`
2. Validate on series from `January 1, 2024` to `December 31, 2024`
3. Select the best model by validation `log_loss`
4. Retrain the selected model on `train + validation`
5. Test once on series from `January 1, 2025` onward

Fallback evaluation logic:

- a percentage-based split still exists for experimentation
- in that mode the latest `20%` of event dates become test
- the earlier block uses expanding-window temporal folds for model selection

This logic is implemented in:

- `src/ml/prematch_baseline.py`

## Training Pipeline

Main implementation:

- `src/ml/prematch_baseline.py`

CLI entrypoint:

- `scripts/train_prematch_baseline.py`

Run locally:

```powershell
python scripts/train_prematch_baseline.py
```

Optional explicit feature path:

```powershell
python scripts/train_prematch_baseline.py --features data/gold/snapshots/20260316T003217Z/match_features_prematch.parquet
```

Optional custom split settings:

```powershell
python scripts/train_prematch_baseline.py --split-strategy fraction --holdout-fraction 0.2 --n-splits 5
```

Explicit calendar split:

```powershell
python scripts/train_prematch_baseline.py --split-strategy calendar --train-end-date 2023-12-31 --validation-start-date 2024-01-01 --validation-end-date 2024-12-31 --test-start-date 2025-01-01
```

## Produced Artifacts

Each training run writes to:

- `data/models/prematch_baseline/<run_id>/`

Files:

- `metrics.json`
- `selection_metrics.parquet`
- `fold_metrics.parquet`
- `holdout_predictions.parquet`
- `league_metrics.parquet`
- `calibration_bins.parquet`
- `<best_model>.pkl`

Latest run pointer:

- `data/models/prematch_baseline/latest_run.json`

## Current Baseline Result

Latest completed run on snapshot `20260316T003217Z`:

- Run ID: `20260316T003506Z`
- Snapshot ID: `20260316T003217Z`
- Feature version: `prematch_v2`
- Best model: `xgboost`

Validation selection metrics:

- `log_loss = 0.5944`
- `brier_score = 0.2058`
- `roc_auc = 0.7426`
- `rows = 513`

Final test metrics:

- `log_loss = 0.6277`
- `brier_score = 0.2157`
- `roc_auc = 0.7099`
- `rows = 703`

Interpretation:

- better than random baseline
- data already contains usable predictive signal across a broader league mix
- expanding the core to `CBLOL/LTA/MSI/WLDs/FST` increased coverage while keeping holdout quality broadly stable
- calibration still needs work before production-grade probability serving

Artifacts for that run:

- `data/models/prematch_baseline/20260316T003506Z/metrics.json`
- `data/models/prematch_baseline/20260316T003506Z/selection_metrics.parquet`
- `data/models/prematch_baseline/20260316T003506Z/fold_metrics.parquet`
- `data/models/prematch_baseline/20260316T003506Z/holdout_predictions.parquet`
- `data/models/prematch_baseline/20260316T003506Z/league_metrics.parquet`
- `data/models/prematch_baseline/20260316T003506Z/calibration_bins.parquet`
- `data/models/prematch_baseline/20260316T003506Z/xgboost.pkl`

## Operational Recommendation

The recommended workflow is:

1. build a new Gold snapshot
2. retrain the prematch baseline
3. compare the new metrics against the previous run
4. only promote a new model if holdout quality improves or remains stable

Recommended default split:

- train: up to `December 31, 2023`
- validation: `January 1, 2024` to `December 31, 2024`
- test: `January 1, 2025` onward

## Next Steps

- improve calibration
- add a persistent upcoming-match scoring ledger
- start collecting live odds snapshots for edge backtesting
- define a future-match scoring pipeline for upcoming series
