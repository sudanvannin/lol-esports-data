"""Train temporal baseline models on Gold prematch features."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from src.ml.prematch_baseline import (
    DEFAULT_SPLIT_STRATEGY,
    DEFAULT_TEST_START_DATE,
    DEFAULT_TRAIN_END_DATE,
    DEFAULT_VALIDATION_END_DATE,
    DEFAULT_VALIDATION_START_DATE,
    run_prematch_baseline,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models on prematch Gold features")
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Optional explicit path to match_features_prematch.parquet",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/models/prematch_baseline"),
        help="Directory where run artifacts will be written",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["calendar", "fraction"],
        default=DEFAULT_SPLIT_STRATEGY,
        help="Evaluation split strategy",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="Fraction of latest event dates reserved for test when using fraction split",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of expanding-window validation folds when using fraction split",
    )
    parser.add_argument(
        "--train-end-date",
        type=str,
        default=DEFAULT_TRAIN_END_DATE,
        help="Inclusive end date for training window when using calendar split",
    )
    parser.add_argument(
        "--validation-start-date",
        type=str,
        default=DEFAULT_VALIDATION_START_DATE,
        help="Inclusive start date for validation window when using calendar split",
    )
    parser.add_argument(
        "--validation-end-date",
        type=str,
        default=DEFAULT_VALIDATION_END_DATE,
        help="Inclusive end date for validation window when using calendar split",
    )
    parser.add_argument(
        "--test-start-date",
        type=str,
        default=DEFAULT_TEST_START_DATE,
        help="Inclusive start date for test window when using calendar split",
    )
    args = parser.parse_args()

    result = run_prematch_baseline(
        features_path=args.features,
        output_root=args.output_root,
        split_strategy=args.split_strategy,
        holdout_fraction=args.holdout_fraction,
        n_splits=args.n_splits,
        train_end_date=args.train_end_date,
        validation_start_date=args.validation_start_date,
        validation_end_date=args.validation_end_date,
        test_start_date=args.test_start_date,
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    print("=" * 72)
    print("PREMATCH BASELINE READY")
    print("=" * 72)
    print(f"run_id: {result.run_id}")
    print(f"snapshot_id: {result.snapshot_id}")
    print(f"features_path: {result.features_path}")
    print(f"output_dir: {result.output_dir}")
    print(f"split_strategy: {metrics['split']['split_strategy']}")
    print(f"best_model: {result.best_model_name}")
    print()
    print("selection_metrics:")
    for item in metrics["selection_metrics"]:
        print(
            "  - "
            f"{item['model_name']}: "
            f"log_loss={item['log_loss']:.4f}, "
            f"brier={item['brier_score']:.4f}, "
            f"auc={item['roc_auc']:.4f}, "
            f"rows={item['evaluation_rows']}"
        )
    print()
    print("test_metrics:")
    for item in metrics["test_metrics"]:
        print(
            "  - "
            f"{item['model_name']}: "
            f"log_loss={item['log_loss']:.4f}, "
            f"brier={item['brier_score']:.4f}, "
            f"auc={item['roc_auc']:.4f}, "
            f"rows={item['rows']}"
        )


if __name__ == "__main__":
    main()
