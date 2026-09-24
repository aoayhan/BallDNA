"""Rebuild the compact app snapshot from the complete local Parquet archive.

Unlike the original compact sync, this keeps every player with recorded minutes.
Small samples remain visible and are handled by reliability labels downstream.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ball_ai.config import settings  # noqa: E402
from ball_ai.data.ingest import BasketballDataset, validate_dataset, write_snapshot  # noqa: E402
from ball_ai.data.sample_loader import load_snapshot_data  # noqa: E402


def _position_label(row: pd.Series) -> str:
    labels = [
        label
        for column, label in (("guard", "G"), ("forward", "F"), ("center", "C"))
        if row.get(column) == 1
    ]
    return "-".join(labels) or "N/A"


def main() -> int:
    """Write one complete season snapshot and replace the local SQLite data."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=settings.nba_season)
    parser.add_argument("--recent-games", type=int, default=settings.recent_games)
    args = parser.parse_args()

    archive = settings.historical_data_dir
    season_stats = pd.read_parquet(archive / "player_season_stats.parquet")
    season_stats = season_stats.loc[
        season_stats["season"].eq(args.season)
        & season_stats["games_played"].gt(0)
        & season_stats["minutes_per_game"].gt(0)
    ].copy()
    if season_stats.empty:
        raise ValueError(f"No player-season rows found for {args.season}.")

    game_parts = list(
        (
            archive
            / "player_game_stats"
            / f"season={args.season}"
            / "season_type=regular"
        ).glob("*.parquet")
    )
    if not game_parts:
        raise FileNotFoundError(f"No regular-season game partition for {args.season}.")
    games = pd.concat([pd.read_parquet(path) for path in game_parts], ignore_index=True)
    games = games.loc[games["player_id"].isin(season_stats["player_id"])].copy()
    latest_team = (
        games.sort_values(["player_id", "game_date", "game_id"])
        .groupby("player_id", as_index=False)
        .tail(1)[["player_id", "team"]]
    )

    biographies = pd.read_parquet(archive / "players.parquet").copy()
    biographies["position"] = biographies.apply(_position_label, axis=1)
    players = (
        season_stats[["player_id", "player_name"]]
        .drop_duplicates("player_id")
        .merge(latest_team, on="player_id", how="left", validate="one_to_one")
        .merge(
            biographies[["player_id", "position"]].drop_duplicates("player_id"),
            on="player_id",
            how="left",
            validate="one_to_one",
        )
    )
    players["team"] = players["team"].fillna("N/A")
    players["position"] = players["position"].fillna("N/A")

    season_columns = [
        "player_id", "season", "games_played", "minutes_per_game",
        "points_per_game", "rebounds_per_game", "assists_per_game",
        "steals_per_game", "blocks_per_game", "turnovers_per_game",
        "field_goal_percentage", "three_point_percentage", "free_throw_percentage",
    ]
    compact_seasons = season_stats[season_columns].copy()
    for column in (
        "field_goal_percentage",
        "three_point_percentage",
        "free_throw_percentage",
    ):
        # No attempts is represented as 0.0 in the compact SQLite schema. The
        # richer Parquet archive retains the distinction as null.
        compact_seasons[column] = compact_seasons[column].fillna(0.0)

    recent = (
        games.sort_values(
            ["player_id", "game_date", "game_id"], ascending=[True, False, False]
        )
        .groupby("player_id", group_keys=False)
        .head(args.recent_games)
        .copy()
    )
    recent["game_id"] = recent["game_id"].astype(str).str.zfill(10)
    game_columns = [
        "game_id", "player_id", "game_date", "opponent", "minutes", "points",
        "rebounds", "assists", "steals", "blocks", "turnovers",
        "field_goal_attempts", "field_goals_made", "three_point_attempts",
        "three_points_made", "free_throw_attempts", "free_throws_made",
    ]
    compact_games = recent[game_columns].copy()

    shots = pd.read_parquet(archive / "player_shot_profiles.parquet")
    shots = shots.loc[
        shots["season"].eq(args.season)
        & shots["player_id"].isin(players["player_id"])
    ].copy()
    missing_shots = set(players["player_id"]) - set(shots["player_id"])
    if missing_shots:
        raise ValueError(f"Shot profiles missing for {len(missing_shots)} players.")

    metadata_path = settings.nba_snapshot_dir / "metadata.json"
    previous_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    metadata = {
        **previous_metadata,
        "season": args.season,
        "snapshot_build": "complete local Parquet archive",
        "player_pool_policy": (
            "All players with at least one regular-season appearance and positive "
            "minutes are included; small samples are exposed through reliability."
        ),
        "snapshot_archive_rebuilt_at": datetime.now(timezone.utc).isoformat(),
    }
    dataset = BasketballDataset(
        players=players.sort_values("player_name").reset_index(drop=True),
        season_stats=compact_seasons.sort_values("player_id").reset_index(drop=True),
        game_stats=compact_games.reset_index(drop=True),
        metadata=metadata,
    )
    validate_dataset(dataset)
    write_snapshot(dataset, settings.nba_snapshot_dir)
    shots.sort_values("player_id").to_csv(
        settings.nba_snapshot_dir / "player_shot_profiles.csv", index=False
    )
    counts = load_snapshot_data(
        settings.nba_snapshot_dir,
        settings.database_path,
        force=True,
    )
    print(
        json.dumps(
            {
                "season": args.season,
                "snapshot_players": len(players),
                "shot_profiles": len(shots),
                **counts,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
