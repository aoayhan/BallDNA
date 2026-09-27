"""Small read-only query helpers for the local historical Parquet archive."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ball_ai.config import settings


def historical_archive_available(root: Path | None = None) -> bool:
    """Return whether the derived season table and coverage manifest exist."""

    archive = Path(root or settings.historical_data_dir)
    return (archive / "player_season_stats.parquet").exists() and (
        archive / "coverage.parquet"
    ).exists()


def get_historical_coverage(root: Path | None = None) -> pd.DataFrame:
    """Return per-season availability for player, team, and shot records."""

    path = Path(root or settings.historical_data_dir) / "coverage.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def get_historical_metadata(root: Path | None = None) -> dict:
    """Return archive provenance and row counts."""

    path = Path(root or settings.historical_data_dir) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def get_latest_historical_player_teams(root: Path | None = None) -> pd.DataFrame:
    """Return each archived player's latest regular-season team and season."""

    archive = Path(root or settings.historical_data_dir)
    paths = sorted(
        archive.glob("player_game_stats/season=*/season_type=regular/*.parquet")
    )
    if not paths:
        return pd.DataFrame(columns=["player_id", "team", "latest_season"])
    games = pd.concat(
        (
            pd.read_parquet(path, columns=["player_id", "team", "game_date"]).assign(
                latest_season=next(
                    part.removeprefix("season=")
                    for part in path.parts
                    if part.startswith("season=")
                )
            )
            for path in paths
        ),
        ignore_index=True,
    )
    games["player_id"] = pd.to_numeric(games["player_id"], errors="coerce")
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["team"] = games["team"].astype("string").str.strip()
    games = games.dropna(subset=["player_id", "game_date", "team"])
    games = games.loc[~games["team"].isin(["", "N/A", "nan"])]
    return (
        games.sort_values("game_date")
        .drop_duplicates("player_id", keep="last")[["player_id", "team", "latest_season"]]
        .assign(player_id=lambda frame: frame["player_id"].astype(int))
        .reset_index(drop=True)
    )


def get_player_season_team(
    player_id: int, season: str, root: Path | None = None
) -> str | None:
    """Return the player's team from their last regular-season game."""

    archive = Path(root or settings.historical_data_dir)
    paths = sorted(
        archive.glob(f"player_game_stats/season={season}/season_type=regular/*.parquet")
    )
    rows = [
        pd.read_parquet(path, columns=["player_id", "team", "game_date"])
        for path in paths
    ]
    if not rows:
        return None
    games = pd.concat(rows, ignore_index=True)
    games = games.loc[pd.to_numeric(games["player_id"], errors="coerce").eq(int(player_id))]
    if games.empty:
        return None
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games = games.dropna(subset=["game_date", "team"]).sort_values("game_date")
    return None if games.empty else str(games.iloc[-1]["team"])


def get_player_season_history(
    player_id: int, root: Path | None = None
) -> pd.DataFrame:
    """Load one player's regular-season history from the compact Parquet table."""

    path = Path(root or settings.historical_data_dir) / "player_season_stats.parquet"
    if not path.exists():
        return pd.DataFrame()
    return (
        pd.read_parquet(path, filters=[("player_id", "=", int(player_id))])
        .sort_values("season")
        .reset_index(drop=True)
    )


def get_player_shot_history(
    player_id: int, root: Path | None = None
) -> pd.DataFrame:
    """Load available shot-style seasons for one player (1996-97 onward)."""

    path = Path(root or settings.historical_data_dir) / "player_shot_profiles.parquet"
    if not path.exists():
        return pd.DataFrame()
    return (
        pd.read_parquet(path, filters=[("player_id", "=", int(player_id))])
        .sort_values("season")
        .reset_index(drop=True)
    )
