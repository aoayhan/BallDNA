"""Normalize one season of NBA shot-detail data from the Apache archive mirror."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ball_ai.analytics.shot_profile import build_shot_profiles


ARCHIVE_DATASET = "cdechoch/nba-data-archive"
ARCHIVE_HOME = f"https://huggingface.co/datasets/{ARCHIVE_DATASET}"
SHOT_COLUMNS = [
    "PLAYER_ID",
    "SHOT_MADE_FLAG",
    "SHOT_DISTANCE",
    "ACTION_TYPE",
    "SHOT_ZONE_BASIC",
]


def season_start_year(season: str) -> int:
    """Convert an NBA season label such as 2025-26 to its archive year."""

    try:
        start = int(season.split("-")[0])
    except (ValueError, IndexError) as exc:
        raise ValueError("Season must look like 2025-26.") from exc
    return start


def shot_snapshot_url(season: str) -> str:
    """Return the season-file download URL used by the refresh script."""

    year = season_start_year(season)
    return (
        f"https://huggingface.co/datasets/{ARCHIVE_DATASET}/resolve/main/"
        f"per_season/shotdetail/{year}.parquet?download=true"
    )


def normalize_shot_snapshot(
    parquet_path: Path,
    season: str,
    eligible_player_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Create compact player fingerprints from a raw shot-detail Parquet file."""

    raw = pd.read_parquet(parquet_path, columns=SHOT_COLUMNS)
    shots = raw.rename(
        columns={
            "PLAYER_ID": "player_id",
            "SHOT_MADE_FLAG": "shot_made",
            "SHOT_DISTANCE": "shot_distance",
            "ACTION_TYPE": "action_type",
            "SHOT_ZONE_BASIC": "shot_zone_basic",
        }
    )
    shots["player_id"] = pd.to_numeric(shots["player_id"], errors="coerce")
    shots = shots.dropna(subset=["player_id"])
    shots["player_id"] = shots["player_id"].astype(int)
    if eligible_player_ids is not None:
        shots = shots.loc[shots["player_id"].isin(eligible_player_ids)]
    profiles = build_shot_profiles(shots)
    profiles.insert(1, "season", season)
    return profiles
