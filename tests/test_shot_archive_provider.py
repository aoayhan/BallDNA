"""Contract tests for the season shot-detail normalizer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ball_ai.data.shot_archive_provider import normalize_shot_snapshot, shot_snapshot_url


def test_archive_url_uses_season_start_year() -> None:
    assert "/2025.parquet" in shot_snapshot_url("2025-26")


def test_normalizer_filters_players_and_builds_profiles(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            (1, 1, 2, "Driving Dunk Shot", "Restricted Area"),
            (1, 0, 25, "Step Back Jump shot", "Above the Break 3"),
            (2, 1, 8, "Driving Floating Jump Shot", "In The Paint (Non-RA)"),
        ],
        columns=[
            "PLAYER_ID", "SHOT_MADE_FLAG", "SHOT_DISTANCE", "ACTION_TYPE",
            "SHOT_ZONE_BASIC",
        ],
    )
    path = tmp_path / "shots.parquet"
    raw.to_parquet(path, index=False)

    profiles = normalize_shot_snapshot(path, "2025-26", {1})

    assert profiles["player_id"].tolist() == [1]
    assert profiles.iloc[0]["season"] == "2025-26"
    assert profiles.iloc[0]["shot_attempts"] == 2
