"""Canonical display order for the season statistics available to BallDNA."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StatDefinition:
    """Describe one statistic's source field, label, and display format."""

    key: str
    abbreviation: str
    label: str
    value_format: str


# Preserve the relative order basketball audiences know from Basketball Reference.
# Fields BallDNA does not currently ingest (Age, GS, ORB, PF, and others) are omitted.
BASKETBALL_REFERENCE_STAT_ORDER = (
    StatDefinition("games_played", "G", "Games played", "integer"),
    StatDefinition("minutes_per_game", "MP", "Minutes per game", "number"),
    StatDefinition("field_goals_made_per_game", "FG", "Field goals made per game", "number"),
    StatDefinition("field_goal_attempts_per_game", "FGA", "Field-goal attempts per game", "number"),
    StatDefinition(
        "field_goal_percentage", "FG%", "Field-goal percentage", "percentage"
    ),
    StatDefinition("three_points_made_per_game", "3P", "Three-pointers made per game", "number"),
    StatDefinition("three_point_attempts_per_game", "3PA", "Three-point attempts per game", "number"),
    StatDefinition(
        "three_point_percentage", "3P%", "Three-point percentage", "percentage"
    ),
    StatDefinition("two_points_made_per_game", "2P", "Two-pointers made per game", "number"),
    StatDefinition("two_point_attempts_per_game", "2PA", "Two-point attempts per game", "number"),
    StatDefinition("two_point_percentage", "2P%", "Two-point percentage", "percentage"),
    StatDefinition("effective_field_goal_percentage", "eFG%", "Effective field-goal percentage", "percentage"),
    StatDefinition(
        "free_throw_percentage", "FT%", "Free-throw percentage", "percentage"
    ),
    StatDefinition("rebounds_per_game", "TRB", "Rebounds per game", "number"),
    StatDefinition("assists_per_game", "AST", "Assists per game", "number"),
    StatDefinition("steals_per_game", "STL", "Steals per game", "number"),
    StatDefinition("blocks_per_game", "BLK", "Blocks per game", "number"),
    StatDefinition("turnovers_per_game", "TOV", "Turnovers per game", "number"),
    StatDefinition("points_per_game", "PTS", "Points per game", "number"),
)


def format_stat_value(value: float | int, value_format: str) -> str:
    """Format a statistic consistently across cards and tables."""

    if value is None:
        return "—"
    try:
        if math.isnan(float(value)):
            return "—"
    except (TypeError, ValueError):
        pass
    if value_format == "percentage":
        return f"{float(value):.1%}"
    if value_format == "integer":
        return f"{int(value)}"
    return f"{float(value):.1f}"
