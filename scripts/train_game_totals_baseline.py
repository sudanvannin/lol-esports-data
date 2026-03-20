"""Train temporal regression baselines for game totals markets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from src.ml.game_totals_baseline import (
    DEFAULT_OUTPUT_ROOT,
    run_game_totals_baseline,
)
from src.ml.prematch_baseline import (
    DEFAULT_TEST_START_DATE,
    DEFAULT_TRAIN_END_DATE,
    DEFAULT_VALIDATION_END_DATE,
    DEFAULT_VALIDATION_START_DATE,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train regression baselines for totals markets")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Optional explicit snapshot directory under data/gold/snapshots",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where run artifacts will be written",
    )
    parser.add_argument(
        "--train-end-date",
        type=str,
        default=DEFAULT_TRAIN_END_DATE,
        help="Inclusive end date for training window",
    )
    parser.add_argument(
        "--validation-start-date",
        type=str,
        default=DEFAULT_VALIDATION_START_DATE,
        help="Inclusive start date for validation window",
    )
    parser.add_argument(
        "--validation-end-date",
        type=str,
        default=DEFAULT_VALIDATION_END_DATE,
        help="Inclusive end date for validation window",
    )
    parser.add_argument(
        "--test-start-date",
        type=str,
        default=DEFAULT_TEST_START_DATE,
        help="Inclusive start date for test window",
    )
    args = parser.parse_args()

    result = run_game_totals_baseline(
        snapshot_dir=args.snapshot_dir,
        output_root=args.output_root,
        train_end_date=args.train_end_date,
        validation_start_date=args.validation_start_date,
        validation_end_date=args.validation_end_date,
        test_start_date=args.test_start_date,
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    print("=" * 72)
    print("GAME TOTALS BASELINE READY")
    print("=" * 72)
    print(f"run_id: {result.run_id}")
    print(f"snapshot_id: {result.snapshot_id}")
    print(f"features_path: {result.features_path}")
    print(f"output_dir: {result.output_dir}")
    print()
    for target, payload in metrics["targets"].items():
        test_metrics = next(
            item for item in metrics["metrics"] if item["target"] == target and item["stage"] == "test"
        )
        print(
            f"{payload['market']}: "
            f"model={payload['best_model_name']}, "
            f"mae={test_metrics['mae']:.4f}, "
            f"rmse={test_metrics['rmse']:.4f}, "
            f"alpha={payload['dispersion_alpha']:.4f}"
        )


if __name__ == "__main__":
    main()
