"""Build the local historical NBA Parquet archive from licensed public sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ball_ai.config import settings  # noqa: E402
from ball_ai.data.historical_archive import (  # noqa: E402
    build_box_score_archive,
    build_players_archive,
    build_shot_archive,
    build_team_box_score_archive,
    write_archive_metadata,
)
from ball_ai.data.kaggle_provider import DATASET_REF, DATASET_URL  # noqa: E402
from ball_ai.data.shot_archive_provider import ARCHIVE_HOME  # noqa: E402


MERGED_SHOTS_URL = (
    "https://huggingface.co/datasets/cdechoch/nba-data-archive/resolve/main/"
    "merged/shotdetail.parquet?download=true"
)


def _download(url: str, destination: Path, timeout: int) -> str:
    """Resume a download when possible and return the completed SHA-256 digest."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        if response.status_code == 416:
            partial.replace(destination)
        else:
            response.raise_for_status()
            mode = "ab" if downloaded and response.status_code == 206 else "wb"
            with partial.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            partial.replace(destination)
    digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kaggle_executable() -> str:
    candidate = Path(sys.executable).parent / "kaggle"
    executable = str(candidate) if candidate.exists() else shutil.which("kaggle")
    if not executable:
        raise RuntimeError("Kaggle CLI is not installed; install the project requirements.")
    return executable


def _download_kaggle_file(filename: str, destination: Path) -> None:
    subprocess.run(
        [
            _kaggle_executable(), "datasets", "download", DATASET_REF,
            "-f", filename, "-p", str(destination), "--unzip", "--force", "--quiet",
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=settings.historical_data_dir)
    parser.add_argument("--raw-dir", type=Path, help="Directory containing Kaggle CSV files.")
    parser.add_argument("--shots-parquet", type=Path, help="Existing merged shot Parquet.")
    parser.add_argument("--skip-box-scores", action="store_true")
    parser.add_argument("--skip-shots", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Retain downloaded CSV/Parquet source copies after successful conversion.",
    )
    args = parser.parse_args()
    if args.skip_box_scores and args.skip_shots:
        parser.error("At least one source must be enabled.")

    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    raw_dir = (args.raw_dir or destination / ".downloads").resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, str] = {}
    box_coverage = None
    team_coverage = None
    shot_coverage = None

    if not args.skip_box_scores:
        statistics = raw_dir / "PlayerStatistics.csv"
        players = raw_dir / "Players.csv"
        team_statistics = raw_dir / "TeamStatistics.csv"
        if not statistics.exists():
            print("Downloading historical player box scores from Kaggle...")
            _download_kaggle_file("PlayerStatistics.csv", raw_dir)
        if not players.exists():
            _download_kaggle_file("Players.csv", raw_dir)
        if not team_statistics.exists():
            _download_kaggle_file("TeamStatistics.csv", raw_dir)
        print("Converting historical box scores to season-partitioned Parquet...")
        build_players_archive(players, destination)
        seasons, box_coverage = build_box_score_archive(
            statistics, destination, chunk_size=args.chunk_size
        )
        team_coverage = build_team_box_score_archive(
            team_statistics, destination, chunk_size=args.chunk_size
        )
        sources["box_scores"] = DATASET_URL
        print(f"Created {len(seasons):,} player-season rows.")

    if not args.skip_shots:
        shot_source = (args.shots_parquet or raw_dir / "shotdetail.parquet").resolve()
        if not shot_source.exists():
            print("Downloading merged shot detail (1996-97 onward)...")
            digest = _download(MERGED_SHOTS_URL, shot_source, args.timeout)
            print(f"Shot archive SHA-256: {digest}")
        print("Building historical shot-style profiles...")
        profiles, shot_coverage = build_shot_archive(shot_source, destination)
        sources["shot_detail"] = ARCHIVE_HOME
        print(f"Created {len(profiles):,} player-season shot profiles.")

    coverage = write_archive_metadata(
        destination,
        box_coverage=box_coverage,
        team_coverage=team_coverage,
        shot_coverage=shot_coverage,
        sources=sources,
    )
    if args.raw_dir is None and not args.keep_downloads and raw_dir.exists():
        shutil.rmtree(raw_dir)
    print(
        json.dumps(
            {
                "archive": str(destination),
                "first_season": str(coverage["season"].min()),
                "last_season": str(coverage["season"].max()),
                "files_size_mb": round(
                    sum(path.stat().st_size for path in destination.rglob("*.parquet"))
                    / 1024 / 1024,
                    1,
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
