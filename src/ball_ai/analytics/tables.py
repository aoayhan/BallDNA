"""Basketball-friendly table preparation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ball_ai.analytics.metrics import add_game_metrics, safe_divide
from ball_ai.analytics.stat_order import (
    BASKETBALL_REFERENCE_STAT_ORDER,
    format_stat_value,
)


GAME_LOG_COLUMNS = [
    "Date", "Opp", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
    "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "TRB", "AST",
    "STL", "BLK", "TOV", "PTS",
]

SEASON_TABLE_IDENTITY_COLUMNS = ["Player", "Season", "Team", "Pos"]

SHOT_PROFILE_COLUMNS = [
    "Season", "Shot Att.", "Rim Freq.", "Paint Freq.", "Midrange Freq.",
    "3P Freq.", "Dunk Freq.", "Layup Freq.", "Floater Freq.",
    "Hook Freq.", "Pull-up Freq.", "Step-back Freq.", "Avg Dist.",
    "Difficulty", "Shot Making +/-",
]

HISTORICAL_SEASON_COLUMNS = [
    "Season", "G", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
    "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB",
    "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
]


def _percentage_strings(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratios = safe_divide(numerator, denominator)
    return pd.Series(
        np.where(denominator.to_numpy(dtype=float) > 0, ratios, np.nan),
        index=numerator.index,
    ).map(lambda value: "—" if pd.isna(value) else f"{value:.1%}")


def basketball_reference_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Return available game fields in familiar Basketball Reference order."""

    if games.empty:
        return pd.DataFrame(columns=GAME_LOG_COLUMNS)

    enriched = add_game_metrics(games)
    two_points_made = enriched["field_goals_made"] - enriched["three_points_made"]
    two_point_attempts = (
        enriched["field_goal_attempts"] - enriched["three_point_attempts"]
    )
    table = pd.DataFrame(
        {
            "Date": enriched["game_date"],
            "Opp": enriched["opponent"],
            "MP": enriched["minutes"].round(1),
            "FG": enriched["field_goals_made"].astype(int),
            "FGA": enriched["field_goal_attempts"].astype(int),
            "FG%": _percentage_strings(
                enriched["field_goals_made"], enriched["field_goal_attempts"]
            ),
            "3P": enriched["three_points_made"].astype(int),
            "3PA": enriched["three_point_attempts"].astype(int),
            "3P%": _percentage_strings(
                enriched["three_points_made"], enriched["three_point_attempts"]
            ),
            "2P": two_points_made.astype(int),
            "2PA": two_point_attempts.astype(int),
            "2P%": _percentage_strings(two_points_made, two_point_attempts),
            "eFG%": _percentage_strings(
                enriched["field_goals_made"]
                + 0.5 * enriched["three_points_made"],
                enriched["field_goal_attempts"],
            ),
            "FT": enriched["free_throws_made"].astype(int),
            "FTA": enriched["free_throw_attempts"].astype(int),
            "FT%": _percentage_strings(
                enriched["free_throws_made"], enriched["free_throw_attempts"]
            ),
            "TRB": enriched["rebounds"].astype(int),
            "AST": enriched["assists"].astype(int),
            "STL": enriched["steals"].astype(int),
            "BLK": enriched["blocks"].astype(int),
            "TOV": enriched["turnovers"].astype(int),
            "PTS": enriched["points"].astype(int),
        }
    )
    return table[GAME_LOG_COLUMNS]


def basketball_reference_season_table(profiles: list[dict]) -> pd.DataFrame:
    """Put one player on each row and season statistics across columns."""

    rows = []
    for profile in profiles:
        row = {
            "Player": str(profile["player_name"]),
            "Season": str(profile["season"]),
            "Team": str(profile["team"]),
            "Pos": str(profile["position"]),
        }
        row.update(
            {
                item.abbreviation: format_stat_value(
                    profile[item.key], item.value_format
                )
                for item in BASKETBALL_REFERENCE_STAT_ORDER
            }
        )
        rows.append(row)

    columns = [
        *SEASON_TABLE_IDENTITY_COLUMNS,
        *(item.abbreviation for item in BASKETBALL_REFERENCE_STAT_ORDER),
    ]
    return pd.DataFrame(rows, columns=columns)


def shot_profile_table(profile: dict | None) -> pd.DataFrame:
    """Render engineered shot features as one horizontal supplementary row."""

    if not profile:
        return pd.DataFrame(columns=SHOT_PROFILE_COLUMNS)
    percentages = {
        "Rim Freq.": "rim_frequency",
        "Paint Freq.": "paint_frequency",
        "Midrange Freq.": "midrange_frequency",
        "3P Freq.": "three_point_frequency",
        "Dunk Freq.": "dunk_frequency",
        "Layup Freq.": "layup_frequency",
        "Floater Freq.": "floater_frequency",
        "Hook Freq.": "hook_frequency",
        "Pull-up Freq.": "pull_up_frequency",
        "Step-back Freq.": "step_back_frequency",
        "Difficulty": "shot_difficulty_index",
        "Shot Making +/-": "shot_making_above_expected",
    }
    row = {
        "Season": str(profile["season"]),
        "Shot Att.": int(profile["shot_attempts"]),
        **{
            label: f"{float(profile[key]):.1%}"
            for label, key in percentages.items()
        },
        "Avg Dist.": f"{float(profile['average_shot_distance']):.1f} ft",
    }
    return pd.DataFrame([row], columns=SHOT_PROFILE_COLUMNS)


def shot_profile_comparison_table(
    profiles: list[tuple[str, dict | None]],
) -> pd.DataFrame:
    """Put players in rows and supplementary shot features in columns."""

    rows = []
    for player_name, profile in profiles:
        table = shot_profile_table(profile)
        if not table.empty:
            rows.append({"Player": player_name, **table.iloc[0].to_dict()})
    return pd.DataFrame(rows, columns=["Player", *SHOT_PROFILE_COLUMNS])


def historical_season_table(history: pd.DataFrame) -> pd.DataFrame:
    """Render career seasons horizontally in Basketball Reference stat order."""

    if history.empty:
        return pd.DataFrame(columns=HISTORICAL_SEASON_COLUMNS)
    mapping = {
        "Season": ("season", "text"),
        "G": ("games_played", "integer"),
        "MP": ("minutes_per_game", "number"),
        "FG": ("field_goals_made_per_game", "number"),
        "FGA": ("field_goal_attempts_per_game", "number"),
        "FG%": ("field_goal_percentage", "percentage"),
        "3P": ("three_points_made_per_game", "number"),
        "3PA": ("three_point_attempts_per_game", "number"),
        "3P%": ("three_point_percentage", "percentage"),
        "2P": ("two_points_made_per_game", "number"),
        "2PA": ("two_point_attempts_per_game", "number"),
        "2P%": ("two_point_percentage", "percentage"),
        "eFG%": ("effective_field_goal_percentage", "percentage"),
        "FT": ("free_throws_made_per_game", "number"),
        "FTA": ("free_throw_attempts_per_game", "number"),
        "FT%": ("free_throw_percentage", "percentage"),
        "ORB": ("offensive_rebounds_per_game", "number"),
        "DRB": ("defensive_rebounds_per_game", "number"),
        "TRB": ("rebounds_per_game", "number"),
        "AST": ("assists_per_game", "number"),
        "STL": ("steals_per_game", "number"),
        "BLK": ("blocks_per_game", "number"),
        "TOV": ("turnovers_per_game", "number"),
        "PF": ("personal_fouls_per_game", "number"),
        "PTS": ("points_per_game", "number"),
    }
    rows: list[dict[str, str]] = []
    for _, season in history.iterrows():
        row: dict[str, str] = {}
        for label, (field, value_format) in mapping.items():
            value = season.get(field)
            row[label] = (
                str(value)
                if value_format == "text"
                else format_stat_value(value, value_format)
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=HISTORICAL_SEASON_COLUMNS)
