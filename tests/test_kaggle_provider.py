"""Contract tests for the CC0 Kaggle data normalizer (no network required)."""

from __future__ import annotations

import pandas as pd

from ball_ai.data.kaggle_provider import KaggleCC0Provider


def test_normalize_aggregates_box_scores_and_positions() -> None:
    raw_games = pd.DataFrame(
        [
            {
                "firstName": "Test", "lastName": "Player", "personId": 101,
                "gameId": 1, "gameDate": "2026-04-01", "gameType": "Regular Season",
                "game_date": "2026-04-01", "numMinutes": 30, "points": 20,
                "assists": 6, "blocks": 1, "steals": 2, "turnovers": 3,
                "reboundsTotal": 8, "fieldGoalsAttempted": 15, "fieldGoalsMade": 8,
                "threePointersAttempted": 5, "threePointersMade": 2,
                "freeThrowsAttempted": 3, "freeThrowsMade": 2,
                "playerteamId": 1610612747, "opponentteamId": 1610612738,
            },
            {
                "firstName": "Test", "lastName": "Player", "personId": 101,
                "gameId": 2, "gameDate": "2026-04-03", "gameType": "Regular Season",
                "game_date": "2026-04-03", "numMinutes": 32, "points": 24,
                "assists": 8, "blocks": 0, "steals": 1, "turnovers": 2,
                "reboundsTotal": 6, "fieldGoalsAttempted": 17, "fieldGoalsMade": 9,
                "threePointersAttempted": 6, "threePointersMade": 3,
                "freeThrowsAttempted": 4, "freeThrowsMade": 3,
                "playerteamId": 1610612747, "opponentteamId": 1610612744,
            },
        ]
    )
    raw_players = pd.DataFrame(
        [{"personId": 101, "guard": 1, "forward": 1, "center": 0}]
    )
    dataset = KaggleCC0Provider(minimum_games=2).normalize(
        raw_games,
        raw_players,
        "2025-26",
        dataset_version="515",
        dataset_updated_at="2026-06-15T11:08:10Z",
    )

    assert dataset.players.iloc[0].to_dict() == {
        "player_id": 101,
        "player_name": "Test Player",
        "team": "LAL",
        "position": "G-F",
    }
    assert dataset.season_stats.iloc[0]["points_per_game"] == 22
    assert dataset.season_stats.iloc[0]["field_goal_percentage"] == 17 / 32
    assert dataset.game_stats.iloc[0]["opponent"] == "GSW"
    assert dataset.metadata["license"] == "CC0-1.0"

