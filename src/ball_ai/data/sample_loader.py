"""Load the bundled demo CSV files into SQLite."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ball_ai.config import settings
from ball_ai.data.database import get_connection, initialize_database, replace_dataset


EXPECTED_COLUMNS = {
    "players.csv": {"player_id", "player_name", "team", "position"},
    "player_season_stats.csv": {
        "player_id", "season", "games_played", "minutes_per_game",
        "points_per_game", "rebounds_per_game", "assists_per_game",
        "steals_per_game", "blocks_per_game", "turnovers_per_game",
        "field_goal_percentage", "three_point_percentage", "free_throw_percentage",
    },
    "player_game_stats.csv": {
        "game_id", "player_id", "game_date", "opponent", "minutes", "points",
        "rebounds", "assists", "steals", "blocks", "turnovers",
        "field_goal_attempts", "field_goals_made", "three_point_attempts",
        "three_points_made", "free_throw_attempts", "free_throws_made",
    },
    "player_shot_profiles.csv": {
        "player_id", "season", "shot_attempts", "shot_makes",
        "three_point_attempts", "three_point_makes", "rim_frequency",
        "paint_frequency", "midrange_frequency", "three_point_frequency",
        "dunk_frequency", "layup_frequency", "floater_frequency",
        "hook_frequency", "pull_up_frequency", "step_back_frequency",
        "average_shot_distance", "shot_difficulty_index",
        "shot_making_above_expected",
    },
}


def _read_and_validate(sample_dir: Path, filename: str) -> pd.DataFrame:
    path = sample_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Required sample file is missing: {path}")
    frame = pd.read_csv(path)
    missing = EXPECTED_COLUMNS[filename] - set(frame.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
    return frame


def load_sample_data(
    database_path: Path | str | None = None,
    sample_dir: Path | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Idempotently load bundled sample data and return row counts."""

    destination = Path(database_path or settings.database_path)
    source = Path(sample_dir or settings.sample_data_dir)
    initialize_database(destination)

    with get_connection(destination) as connection:
        existing = connection.execute("SELECT COUNT(*) FROM players").fetchone()[0]
    if existing and not force:
        return {"players": existing, "season_rows": 0, "game_rows": 0}

    players = _read_and_validate(source, "players.csv")
    seasons = _read_and_validate(source, "player_season_stats.csv")
    games = _read_and_validate(source, "player_game_stats.csv")

    return replace_dataset(
        players,
        seasons,
        games,
        metadata={
            "provider": "Bundled illustrative sample",
            "provider_kind": "demo",
            "season": str(seasons["season"].max()),
            "retrieved_at": "not applicable",
            "source_url": "data/sample/README.md",
            "recent_game_provenance": "Deterministically generated illustrative rows",
        },
        database_path=destination,
    )


def load_snapshot_data(
    snapshot_dir: Path,
    database_path: Path | str | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Load a normalized, provenance-bearing snapshot created by the sync script."""

    destination = Path(database_path or settings.database_path)
    initialize_database(destination)
    with get_connection(destination) as connection:
        existing = connection.execute("SELECT COUNT(*) FROM players").fetchone()[0]
    if existing and not force:
        return {"players": existing, "season_rows": 0, "game_rows": 0}

    players = _read_and_validate(snapshot_dir, "players.csv")
    seasons = _read_and_validate(snapshot_dir, "player_season_stats.csv")
    games = _read_and_validate(snapshot_dir, "player_game_stats.csv")
    shot_path = snapshot_dir / "player_shot_profiles.csv"
    shot_profiles = (
        _read_and_validate(snapshot_dir, "player_shot_profiles.csv")
        if shot_path.exists()
        else None
    )
    metadata_path = snapshot_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Snapshot metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return replace_dataset(
        players,
        seasons,
        games,
        metadata=metadata,
        database_path=destination,
        shot_profiles=shot_profiles,
    )


def ensure_sample_database(database_path: Path | str | None = None) -> None:
    """Load the licensed snapshot when available, otherwise use demo data."""

    destination = Path(database_path or settings.database_path)
    snapshot = settings.nba_snapshot_dir
    if (snapshot / "metadata.json").exists():
        load_snapshot_data(snapshot, destination, force=False)
    else:
        load_sample_data(destination, force=False)
