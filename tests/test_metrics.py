"""Tests for basketball metric calculations."""

from __future__ import annotations

import pandas as pd
import pytest

from ball_ai.analytics.metrics import (
    add_game_metrics,
    effective_field_goal_percentage,
    safe_divide,
    true_shooting_percentage,
)


def test_safe_divide_handles_zero_denominator() -> None:
    assert safe_divide(8, 0) == 0


def test_effective_field_goal_percentage() -> None:
    assert effective_field_goal_percentage(8, 4, 20) == pytest.approx(0.5)


def test_true_shooting_percentage() -> None:
    assert true_shooting_percentage(25, 18, 6) == pytest.approx(25 / (2 * (18 + .44 * 6)))


def test_add_game_metrics_does_not_mutate_input() -> None:
    games = pd.DataFrame(
        {
            "points": [20], "field_goals_made": [7], "field_goal_attempts": [15],
            "three_points_made": [2], "free_throw_attempts": [5],
            "assists": [6], "turnovers": [2],
        }
    )
    enriched = add_game_metrics(games)
    assert "true_shooting_percentage" not in games
    assert enriched.loc[0, "assist_turnover_ratio"] == 3

