"""Read-model tests for the local Parquet archive."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ball_ai.data.historical_store import (
    get_historical_coverage,
    get_latest_historical_player_teams,
    get_player_season_history,
    historical_archive_available,
)


def test_historical_queries_filter_player_without_loading_game_archive(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        [
            {"player_id": 1, "season": "2023-24", "points_per_game": 10},
            {"player_id": 1, "season": "2024-25", "points_per_game": 12},
            {"player_id": 2, "season": "2024-25", "points_per_game": 20},
        ]
    ).to_parquet(tmp_path / "player_season_stats.parquet", index=False)
    pd.DataFrame(
        [{"season": "2024-25", "season_type": "regular", "box_score_rows": 3}]
    ).to_parquet(tmp_path / "coverage.parquet", index=False)

    assert historical_archive_available(tmp_path)
    assert get_player_season_history(1, tmp_path)["season"].tolist() == [
        "2023-24", "2024-25"
    ]
    assert len(get_historical_coverage(tmp_path)) == 1


def test_latest_historical_team_uses_the_last_regular_season_game(tmp_path: Path) -> None:
    directory = (
        tmp_path / "player_game_stats" / "season=2024-25" / "season_type=regular"
    )
    directory.mkdir(parents=True)
    pd.DataFrame([
        {"player_id": 1, "team": "LAL", "game_date": "2025-01-01"},
        {"player_id": 1, "team": "DAL", "game_date": "2025-02-01"},
        {"player_id": 2, "team": "BOS", "game_date": "2025-01-15"},
    ]).to_parquet(directory / "part.parquet", index=False)

    result = get_latest_historical_player_teams(tmp_path).set_index("player_id")
    assert result.loc[1, "team"] == "DAL"
    assert result.loc[1, "latest_season"] == "2024-25"
    assert result.loc[2, "team"] == "BOS"
