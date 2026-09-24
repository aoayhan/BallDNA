"""Tests for Basketball Reference-style presentation tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ball_ai.analytics.stat_order import BASKETBALL_REFERENCE_STAT_ORDER
from ball_ai.analytics.tables import (
    GAME_LOG_COLUMNS,
    basketball_reference_game_log,
    basketball_reference_season_table,
    historical_season_table,
    shot_profile_comparison_table,
)
from ball_ai.data.database import get_player_profile, get_players, get_recent_games


def test_game_log_uses_familiar_column_order(demo_database: Path) -> None:
    player_id = int(get_players(demo_database).iloc[0].player_id)
    table = basketball_reference_game_log(
        get_recent_games(player_id, 1, demo_database)
    )
    assert table.columns.tolist() == GAME_LOG_COLUMNS
    assert table.iloc[0]["2P"] == table.iloc[0]["FG"] - table.iloc[0]["3P"]


def test_season_stats_use_familiar_relative_order() -> None:
    assert [item.abbreviation for item in BASKETBALL_REFERENCE_STAT_ORDER] == [
        "G", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA",
        "2P%", "eFG%", "FT%", "TRB", "AST", "STL", "BLK", "TOV", "PTS"
    ]


def test_comparison_table_places_players_in_rows_and_stats_in_columns(
    demo_database: Path,
) -> None:
    players = get_players(demo_database).head(2)
    profiles = [
        get_player_profile(int(row.player_id), demo_database)
        for row in players.itertuples()
    ]
    table = basketball_reference_season_table(profiles)

    assert table.shape[0] == 2
    assert table.columns.tolist() == [
        "Player", "Season", "Team", "Pos", "G", "MP", "FG", "FGA", "FG%",
        "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%", "FT%", "TRB",
        "AST", "STL", "BLK", "TOV", "PTS",
    ]


def test_shot_comparison_places_players_in_rows() -> None:
    profile = {
        "season": "2025-26",
        "shot_attempts": 100,
        "rim_frequency": .2,
        "paint_frequency": .1,
        "midrange_frequency": .2,
        "three_point_frequency": .5,
        "dunk_frequency": .02,
        "layup_frequency": .18,
        "floater_frequency": .05,
        "hook_frequency": .01,
        "pull_up_frequency": .2,
        "step_back_frequency": .1,
        "average_shot_distance": 16.4,
        "shot_difficulty_index": .56,
        "shot_making_above_expected": .02,
    }
    table = shot_profile_comparison_table([("Player A", profile), ("Player B", profile)])

    assert table.shape[0] == 2
    assert table.columns[0] == "Player"
    assert table.iloc[0]["3P Freq."] == "50.0%"


def test_historical_table_uses_basketball_reference_order() -> None:
    history = pd.DataFrame(
        [{
            "season": "1996-97", "games_played": 10, "minutes_per_game": 20,
            "field_goals_made_per_game": 4, "field_goal_attempts_per_game": 9,
            "field_goal_percentage": .444, "three_points_made_per_game": 1,
            "three_point_attempts_per_game": 3, "three_point_percentage": .333,
            "two_points_made_per_game": 3, "two_point_attempts_per_game": 6,
            "two_point_percentage": .5, "effective_field_goal_percentage": .5,
            "free_throws_made_per_game": 2, "free_throw_attempts_per_game": 3,
            "free_throw_percentage": .667, "offensive_rebounds_per_game": 1,
            "defensive_rebounds_per_game": 3, "rebounds_per_game": 4,
            "assists_per_game": 5, "steals_per_game": 1, "blocks_per_game": .5,
            "turnovers_per_game": 2, "personal_fouls_per_game": 2.5,
            "points_per_game": 11,
        }]
    )

    table = historical_season_table(history)

    assert table.columns.tolist() == [
        "Season", "G", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
        "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB",
        "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
    ]
