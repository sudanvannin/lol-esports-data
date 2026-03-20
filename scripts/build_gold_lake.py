"""Build the Gold lake snapshot and print a compact summary."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from src.transform.gold_layer import GoldLayerBuilder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> None:
    builder = GoldLayerBuilder()
    result = builder.build()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    latest_pointer = Path("data/gold/latest_snapshot.json")

    print("=" * 72)
    print("GOLD SNAPSHOT READY")
    print("=" * 72)
    print(f"snapshot_id: {result.snapshot_id}")
    print(f"snapshot_dir: {result.snapshot_dir}")
    print(f"latest_pointer: {latest_pointer}")
    print()
    print("tables:")
    for name, path in sorted(result.tables.items()):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  - {name}: {path.name} ({size_mb:.2f} MB)")

    print()
    print("row_counts:")
    for table_name, row_count in sorted(manifest["table_counts"].items()):
        print(f"  - {table_name}: {row_count:,}")

    print()
    print("validation:")
    print(f"  - total_checks: {manifest['validation']['total_checks']}")
    print(f"  - failed_checks: {manifest['validation']['failed_checks']}")
    if manifest["validation"]["failed_check_names"]:
        for check_name in manifest["validation"]["failed_check_names"]:
            print(f"    * {check_name}")


if __name__ == "__main__":
    main()
