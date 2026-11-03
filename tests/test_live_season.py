from datetime import date

import pandas as pd

from ball_ai.data.live_season import build_live_offensive_features, current_nba_season


def test_current_nba_season_changes_in_september() -> None:
    assert current_nba_season(date(2026, 8, 31)) == "2025-26"
    assert current_nba_season(date(2026, 9, 1)) == "2026-27"


def test_live_floor_keeps_three_game_player_and_rejects_two_games() -> None:
    stats = pd.DataFrame([
        {
            "PLAYER_ID": player_id, "PLAYER_NAME": name, "GP": games, "MIN": 60,
            "FGM": 8, "FGA": 15, "FG3M": 2, "FG3A": 5, "FTM": 3, "FTA": 4,
            "OREB": 1, "DREB": 5, "REB": 6, "AST": 4, "STL": 1, "BLK": 0,
            "TOV": 2, "PF": 3, "PTS": 21, "PLUS_MINUS": 2,
        }
        for player_id, name, games in ((1, "Eligible Player", 3), (2, "Too Few Games", 2))
    ])
    bios = pd.DataFrame({"PLAYER_ID": [1, 2], "AGE": [24, 25]})
    shots = pd.DataFrame([
        {
            "PLAYER_ID": player_id,
            "SHOT_MADE_FLAG": int(index % 2 == 0),
            "SHOT_DISTANCE": 24 if index < 5 else 3,
            "ACTION_TYPE": "Pullup Jump shot" if index < 5 else "Layup Shot",
            "SHOT_ZONE_BASIC": "Above the Break 3" if index < 5 else "Restricted Area",
        }
        for player_id in (1, 2)
        for index in range(15)
    ])
    historical_bios = pd.DataFrame({
        "player_id": [1, 2], "birthDate": ["2002-01-01", "2001-01-01"],
        "guard": [1, 1], "forward": [0, 0], "center": [0, 0],
        "heightInches": [76, 75], "bodyWeightLbs": [195, 190],
    })

    features = build_live_offensive_features(
        stats, bios, shots, historical_bios, season="2026-27"
    ).set_index("player_id")

    assert bool(features.loc[1, "offensive_eligible"])
    assert not bool(features.loc[2, "offensive_eligible"])
    assert features.loc[1, "pull_up_frequency"] > 0
