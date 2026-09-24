"""Reusable basketball metric calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator, denominator):
    """Divide values while returning zero for missing or zero denominators."""

    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    result = np.divide(
        numerator_array,
        denominator_array,
        out=np.zeros_like(numerator_array, dtype=float),
        where=denominator_array != 0,
    )
    return float(result) if result.ndim == 0 else result


def effective_field_goal_percentage(fgm, three_pm, fga):
    """Calculate eFG%, which gives an extra half-make for three-pointers."""

    return safe_divide(np.asarray(fgm) + 0.5 * np.asarray(three_pm), fga)


def true_shooting_percentage(points, fga, fta):
    """Estimate scoring efficiency across field goals and free throws."""

    return safe_divide(points, 2 * (np.asarray(fga) + 0.44 * np.asarray(fta)))


def add_game_metrics(games: pd.DataFrame) -> pd.DataFrame:
    """Add derived efficiency features without mutating the input frame."""

    enriched = games.copy()
    if enriched.empty:
        enriched["effective_field_goal_percentage"] = pd.Series(dtype=float)
        enriched["true_shooting_percentage"] = pd.Series(dtype=float)
        enriched["assist_turnover_ratio"] = pd.Series(dtype=float)
        return enriched
    enriched["effective_field_goal_percentage"] = effective_field_goal_percentage(
        enriched["field_goals_made"],
        enriched["three_points_made"],
        enriched["field_goal_attempts"],
    )
    enriched["true_shooting_percentage"] = true_shooting_percentage(
        enriched["points"],
        enriched["field_goal_attempts"],
        enriched["free_throw_attempts"],
    )
    enriched["assist_turnover_ratio"] = safe_divide(
        enriched["assists"], enriched["turnovers"]
    )
    return enriched


def summarize_games(games: pd.DataFrame) -> dict[str, float | int]:
    """Create a compact aggregate used by both UI and AI grounding."""

    enriched = add_game_metrics(games)
    if enriched.empty:
        return {"games": 0}
    return {
        "games": int(len(enriched)),
        "minutes": round(float(enriched["minutes"].mean()), 1),
        "points": round(float(enriched["points"].mean()), 1),
        "rebounds": round(float(enriched["rebounds"].mean()), 1),
        "assists": round(float(enriched["assists"].mean()), 1),
        "steals": round(float(enriched["steals"].mean()), 1),
        "blocks": round(float(enriched["blocks"].mean()), 1),
        "turnovers": round(float(enriched["turnovers"].mean()), 1),
        "effective_field_goal_percentage": round(
            float(enriched["effective_field_goal_percentage"].mean()), 3
        ),
        "true_shooting_percentage": round(
            float(enriched["true_shooting_percentage"].mean()), 3
        ),
        "assist_turnover_ratio": round(
            float(enriched["assist_turnover_ratio"].mean()), 2
        ),
    }

