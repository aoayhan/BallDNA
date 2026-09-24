"""Initialize or refresh the local BallDNA SQLite database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ball_ai.config import settings  # noqa: E402
from ball_ai.data.sample_loader import load_sample_data, load_snapshot_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace existing demo rows.")
    args = parser.parse_args()
    if (settings.nba_snapshot_dir / "metadata.json").exists():
        counts = load_snapshot_data(
            settings.nba_snapshot_dir, settings.database_path, force=args.force
        )
        source = settings.nba_snapshot_dir
    else:
        counts = load_sample_data(settings.database_path, force=args.force)
        source = settings.sample_data_dir
    print(f"Database ready at {settings.database_path}")
    print(f"Source: {source}")
    print(counts)


if __name__ == "__main__":
    main()
