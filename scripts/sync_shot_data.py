"""Refresh the compact NBA shot-style profiles used by similarity retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ball_ai.config import settings  # noqa: E402
from ball_ai.data.sample_loader import load_snapshot_data  # noqa: E402
from ball_ai.data.shot_archive_provider import (  # noqa: E402
    ARCHIVE_HOME,
    normalize_shot_snapshot,
    shot_snapshot_url,
)


def _download(url: str, destination: Path, timeout: int) -> str:
    """Stream a remote file and return its SHA-256 digest."""

    digest = hashlib.sha256()
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    digest.update(chunk)
                    handle.write(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=settings.nba_season)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--parquet",
        type=Path,
        help="Use an existing season Parquet file instead of downloading it.",
    )
    args = parser.parse_args()

    snapshot = settings.nba_snapshot_dir
    players_path = snapshot / "players.csv"
    metadata_path = snapshot / "metadata.json"
    if not players_path.exists() or not metadata_path.exists():
        print("Base NBA snapshot is missing; run scripts/sync_nba_data.py first.")
        return 1

    players = pd.read_csv(players_path)
    eligible_ids = set(players["player_id"].astype(int))
    url = shot_snapshot_url(args.season)
    try:
        if args.parquet:
            parquet_path = args.parquet
            digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
            profiles = normalize_shot_snapshot(parquet_path, args.season, eligible_ids)
        else:
            with tempfile.TemporaryDirectory(prefix="ball_ai_shots_") as temp:
                parquet_path = Path(temp) / "shotdetail.parquet"
                print(f"Downloading shot detail for {args.season}...")
                digest = _download(url, parquet_path, args.timeout)
                profiles = normalize_shot_snapshot(
                    parquet_path, args.season, eligible_ids
                )
    except (OSError, ValueError, requests.RequestException) as exc:
        print(f"Shot refresh failed safely; existing data was not changed.\n{exc}")
        return 1

    missing = eligible_ids - set(profiles["player_id"].astype(int))
    if missing:
        print(f"Shot refresh rejected: {len(missing)} app players have no shot profile.")
        return 1

    profile_path = snapshot / "player_shot_profiles.csv"
    profile_temp = snapshot / ".player_shot_profiles.csv.tmp"
    profiles.to_csv(profile_temp, index=False)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "shot_data_provider": "NBA Data Archive on Hugging Face",
            "shot_data_license": "Apache-2.0 (archive declaration)",
            "shot_data_source_url": ARCHIVE_HOME,
            "shot_data_file_url": url,
            "shot_data_retrieved_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "shot_data_sha256": digest,
            "shot_data_event_rows": str(int(profiles["shot_attempts"].sum())),
            "shot_profile_rows": str(len(profiles)),
            "shot_feature_note": (
                "Player shot mix and expected-FG features engineered locally from "
                "NBA shot-detail action, zone, distance, and make/miss fields."
            ),
        }
    )
    metadata_temp = snapshot / ".metadata.json.tmp"
    metadata_temp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    profile_temp.replace(profile_path)
    metadata_temp.replace(metadata_path)

    counts = load_snapshot_data(snapshot, settings.database_path, force=True)
    print(json.dumps({"shot_profiles": len(profiles), **counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
