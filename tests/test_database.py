"""Integration tests for CSV-to-SQLite loading and read models."""

from __future__ import annotations

from pathlib import Path

from ball_ai.data.database import (
    get_data_metadata,
    get_player_profile,
    get_players,
    get_recent_games,
    get_season_stats,
)
from ball_ai.analytics.similarity import find_similar_players


def test_sample_loader_populates_queryable_database(demo_database: Path) -> None:
    players = get_players(demo_database)
    assert len(players) == 12
    profile = get_player_profile(int(players.iloc[0].player_id), demo_database)
    assert profile is not None
    assert profile["games_played"] > 0
    games = get_recent_games(int(players.iloc[0].player_id), 10, demo_database)
    assert len(games) == 10
    assert games["game_date"].is_monotonic_increasing
    metadata = get_data_metadata(demo_database)
    assert metadata["provider_kind"] == "demo"


def test_database_profiles_feed_similarity_without_duplicate_ids(
    demo_database: Path,
) -> None:
    players = get_players(demo_database)
    season = get_season_stats(demo_database)
    assert list(season.columns).count("player_id") == 1

    player_id = int(players.iloc[0].player_id)
    neighbors = find_similar_players(season, player_id, top_n=3)
    assert len(neighbors) == 3
    assert player_id not in neighbors["player_id"].tolist()
