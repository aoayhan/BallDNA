"""Extensible ingestion boundary for future live providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd


@dataclass
class BasketballDataset:
    """Normalized frames expected by the persistence layer."""

    players: pd.DataFrame
    season_stats: pd.DataFrame
    game_stats: pd.DataFrame
    metadata: dict[str, str] = field(default_factory=dict)


class BasketballDataProvider(Protocol):
    """Contract implemented by each replaceable basketball data source."""

    def fetch(self, season: str) -> BasketballDataset:
        """Fetch and normalize player, season, and recent-game data."""
        ...


def validate_dataset(dataset: BasketballDataset) -> None:
    """Fail fast before external data is written to the application database."""

    if dataset.players.empty:
        raise ValueError("The provider returned no players.")
    if dataset.season_stats.empty:
        raise ValueError("The provider returned no season statistics.")
    if dataset.game_stats.empty:
        raise ValueError("The provider returned no recent game statistics.")
    if not set(dataset.season_stats.player_id).issubset(set(dataset.players.player_id)):
        raise ValueError("Season statistics contain unknown player IDs.")
    if not set(dataset.game_stats.player_id).issubset(set(dataset.players.player_id)):
        raise ValueError("Game statistics contain unknown player IDs.")
    if dataset.players.player_id.duplicated().any():
        raise ValueError("Player metadata contains duplicate player IDs.")
    if dataset.season_stats[["player_id", "season"]].duplicated().any():
        raise ValueError("Season statistics contain duplicate player-season rows.")
    if dataset.game_stats[["game_id", "player_id"]].duplicated().any():
        raise ValueError("Game statistics contain duplicate player-game rows.")


def write_snapshot(dataset: BasketballDataset, destination: Path) -> None:
    """Write normalized, provenance-bearing CSVs for a reproducible offline demo."""

    import json

    destination.mkdir(parents=True, exist_ok=True)
    dataset.players.to_csv(destination / "players.csv", index=False)
    dataset.season_stats.to_csv(destination / "player_season_stats.csv", index=False)
    dataset.game_stats.to_csv(destination / "player_game_stats.csv", index=False)
    metadata = {
        **dataset.metadata,
        "snapshot_written_at": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
