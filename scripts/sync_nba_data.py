"""Download the public CC0 Kaggle source and refresh the BallDNA snapshot."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ball_ai.config import settings  # noqa: E402
from ball_ai.data.database import replace_dataset  # noqa: E402
from ball_ai.data.ingest import write_snapshot  # noqa: E402
from ball_ai.data.kaggle_provider import (  # noqa: E402
    DATASET_REF,
    KaggleCC0Provider,
    KaggleDatasetError,
)


def _dataset_metadata(timeout: int) -> dict:
    response = requests.get(
        f"https://www.kaggle.com/api/v1/datasets/view/{DATASET_REF}", timeout=timeout
    )
    response.raise_for_status()
    metadata = response.json()
    if metadata.get("licenseName") != "CC0: Public Domain":
        raise KaggleDatasetError(
            f"Expected CC0 source license, received: {metadata.get('licenseName')}"
        )
    return metadata


def _download_file(filename: str, destination: Path) -> None:
    local_cli = Path(sys.executable).parent / "kaggle"
    executable = str(local_cli) if local_cli.exists() else shutil.which("kaggle")
    if not executable:
        raise KaggleDatasetError("Kaggle CLI is not installed.")
    command = [
        executable, "datasets", "download", DATASET_REF,
        "-f", filename, "-p", str(destination), "--unzip", "--force", "--quiet",
    ]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=settings.nba_season, help="NBA season, e.g. 2025-26")
    parser.add_argument("--recent-games", type=int, default=settings.recent_games)
    parser.add_argument("--minimum-games", type=int, default=settings.minimum_games)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Use existing PlayerStatistics.csv and Players.csv instead of downloading.",
    )
    args = parser.parse_args()

    try:
        metadata = _dataset_metadata(args.timeout)
        provider = KaggleCC0Provider(
            recent_games=args.recent_games, minimum_games=args.minimum_games
        )
        if args.raw_dir:
            dataset = provider.from_files(
                args.raw_dir / "PlayerStatistics.csv",
                args.raw_dir / "Players.csv",
                args.season,
                str(metadata["currentVersionNumber"]),
                metadata["lastUpdated"],
            )
        else:
            with tempfile.TemporaryDirectory(prefix="hooplens_kaggle_") as temp:
                raw_dir = Path(temp)
                print("Downloading the two required CC0 files from Kaggle...")
                _download_file("PlayerStatistics.csv", raw_dir)
                _download_file("Players.csv", raw_dir)
                dataset = provider.from_files(
                    raw_dir / "PlayerStatistics.csv",
                    raw_dir / "Players.csv",
                    args.season,
                    str(metadata["currentVersionNumber"]),
                    metadata["lastUpdated"],
                )
    except (KaggleDatasetError, requests.RequestException, subprocess.CalledProcessError) as exc:
        print(f"Refresh failed safely; the existing database was not changed.\n{exc}")
        return 1

    write_snapshot(dataset, settings.nba_snapshot_dir)
    counts = replace_dataset(
        dataset.players,
        dataset.season_stats,
        dataset.game_stats,
        dataset.metadata,
        settings.database_path,
    )
    print(json.dumps({"snapshot": str(settings.nba_snapshot_dir), **counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
