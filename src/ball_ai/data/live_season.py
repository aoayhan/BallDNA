"""Normalize an ongoing NBA season for provisional Player DNA inference."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ball_ai.analytics.play_style import build_style_features
from ball_ai.analytics.shot_profile import build_shot_profiles


LIVE_MIN_GAMES = 3
LIVE_MIN_SHOTS = 15


def current_nba_season(today: date | None = None) -> str:
    """Return the season containing ``today`` using the NBA's start year."""

    value = today or date.today()
    start = value.year if value.month >= 9 else value.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.Series(
        np.divide(
            pd.to_numeric(numerator, errors="coerce").fillna(0),
            pd.to_numeric(denominator, errors="coerce").fillna(0),
            out=np.zeros(len(numerator), dtype=float),
            where=pd.to_numeric(denominator, errors="coerce").fillna(0).ne(0),
        ),
        index=numerator.index,
    )


def build_live_offensive_features(
    player_stats: pd.DataFrame,
    player_bios: pd.DataFrame,
    shots: pd.DataFrame,
    historical_bios: pd.DataFrame,
    *,
    season: str,
    minimum_games: int = LIVE_MIN_GAMES,
    minimum_shots: int = LIVE_MIN_SHOTS,
) -> pd.DataFrame:
    """Build frozen-model inputs while applying a deliberately small live floor."""

    stat_columns = {
        "PLAYER_ID", "PLAYER_NAME", "GP", "MIN", "FGM", "FGA", "FG3M", "FG3A",
        "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
    }
    shot_columns = {
        "PLAYER_ID", "SHOT_MADE_FLAG", "SHOT_DISTANCE", "ACTION_TYPE", "SHOT_ZONE_BASIC",
    }
    if missing := stat_columns - set(player_stats.columns):
        raise ValueError(f"Live player stats are missing columns: {sorted(missing)}")
    if missing := shot_columns - set(shots.columns):
        raise ValueError(f"Live shot data is missing columns: {sorted(missing)}")
    if minimum_games < 1 or minimum_shots < 1:
        raise ValueError("Live eligibility limits must be positive.")

    stats = player_stats.copy()
    numeric = stat_columns - {"PLAYER_NAME"}
    for column in numeric:
        stats[column] = pd.to_numeric(stats[column], errors="coerce").fillna(0)
    stats = stats.loc[stats["GP"].gt(0)].copy()
    games = stats["GP"]
    fgm, fga = stats["FGM"], stats["FGA"]
    three_made, three_attempted = stats["FG3M"], stats["FG3A"]
    ftm, fta = stats["FTM"], stats["FTA"]
    player_seasons = pd.DataFrame({
        "player_id": stats["PLAYER_ID"].astype(int),
        "player_name": stats["PLAYER_NAME"].astype(str),
        "season": season,
        "games_played": games.astype(int),
        "field_goals_made_total": fgm,
        "field_goal_attempts_total": fga,
        "three_points_made_total": three_made,
        "three_point_attempts_total": three_attempted,
        "free_throws_made_total": ftm,
        "free_throw_attempts_total": fta,
        "offensive_rebounds_total": stats["OREB"],
        "defensive_rebounds_total": stats["DREB"],
        "rebounds_total": stats["REB"],
        "assists_total": stats["AST"],
        "steals_total": stats["STL"],
        "blocks_total": stats["BLK"],
        "turnovers_total": stats["TOV"],
        "personal_fouls_total": stats["PF"],
        "points_total": stats["PTS"],
        "plus_minus_total": pd.to_numeric(stats.get("PLUS_MINUS"), errors="coerce"),
        "minutes_per_game": _ratio(stats["MIN"], games),
        "field_goal_percentage": _ratio(fgm, fga),
        "three_point_percentage": _ratio(three_made, three_attempted),
        "free_throw_percentage": _ratio(ftm, fta),
        "two_points_made_total": fgm - three_made,
        "two_point_attempts_total": fga - three_attempted,
    })

    normalized_shots = shots.rename(columns={
        "PLAYER_ID": "player_id",
        "SHOT_MADE_FLAG": "shot_made",
        "SHOT_DISTANCE": "shot_distance",
        "ACTION_TYPE": "action_type",
        "SHOT_ZONE_BASIC": "shot_zone_basic",
    })
    shot_profiles = build_shot_profiles(normalized_shots)
    shot_profiles["season"] = season

    bio_columns = [
        "player_id", "birthDate", "guard", "forward", "center", "heightInches", "bodyWeightLbs",
    ]
    biographies = historical_bios.reindex(columns=bio_columns).drop_duplicates("player_id").copy()
    missing_ids = set(player_seasons["player_id"]) - set(biographies["player_id"])
    if missing_ids:
        biographies = pd.concat([
            biographies,
            pd.DataFrame({
                "player_id": sorted(missing_ids), "birthDate": pd.NaT,
                "guard": 0, "forward": 0, "center": 0,
                "heightInches": np.nan, "bodyWeightLbs": np.nan,
            }),
        ], ignore_index=True)

    features = build_style_features(player_seasons, shot_profiles, biographies)
    if "three_point_frequency" not in features:
        features = features.merge(
            shot_profiles[["player_id", "season", "three_point_frequency"]],
            on=["player_id", "season"],
            how="left",
            validate="one_to_one",
        )
    age_by_id = (
        player_bios.assign(PLAYER_ID=pd.to_numeric(player_bios["PLAYER_ID"], errors="coerce"))
        .dropna(subset=["PLAYER_ID"])
        .drop_duplicates("PLAYER_ID")
        .set_index("PLAYER_ID")["AGE"]
    )
    features["age"] = features["player_id"].map(age_by_id)
    for column in ("dpm", "o_dpm", "d_dpm"):
        features[column] = np.nan
    games = pd.to_numeric(features["games_played"], errors="coerce").fillna(0)
    attempts = pd.to_numeric(features["shot_attempts"], errors="coerce").fillna(0)
    features["offensive_eligible"] = games.ge(minimum_games) & attempts.ge(minimum_shots)
    features["offensive_reliability"] = np.minimum(games / 20, attempts / 400).clip(0, 1)
    return features.sort_values("player_name").reset_index(drop=True)
