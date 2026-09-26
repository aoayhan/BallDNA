"""Checks for offline reliability and feature-group challengers."""

import numpy as np
import pandas as pd

from scripts.evaluate_reliability_smoothing import reliability_smooth
from scripts.evaluate_grouped_behavior import FEATURE_GROUPS, group_feature_weights
from scripts.train_multiview_temporal_challenger import temporal_targets


def test_reliability_smoothing_shrinks_small_samples_more() -> None:
    rows = pd.DataFrame({
        "season": ["2024-25"] * 3,
        "shot_attempts": [10, 100, 1_000],
        "field_goal_attempts_total": [10, 100, 1_000],
        **{
            column: [1.0, 0.5, 0.5]
            for column in [
                "rim_frequency", "paint_frequency", "midrange_frequency",
                "three_point_frequency", "dunk_frequency", "layup_frequency",
                "floater_frequency", "hook_frequency", "pull_up_frequency",
                "step_back_frequency", "three_point_attempt_rate",
                "free_throw_attempt_rate",
            ]
        },
    })

    smoothed = reliability_smooth(rows, 50)

    assert abs(smoothed.loc[0, "rim_frequency"] - 0.5) > abs(
        smoothed.loc[1, "rim_frequency"] - 0.5
    )
    assert abs(smoothed.loc[0, "rim_frequency"] - 1.0) > abs(
        smoothed.loc[2, "rim_frequency"] - 0.5
    )


def test_group_weights_match_requested_squared_distance_budgets() -> None:
    budgets = {
        "shot_location": .25, "shot_action": .25, "creation": .20,
        "playmaking": .25, "offensive_rebounding": .05,
    }
    weights = group_feature_weights(budgets)
    group_energy = {
        group: sum(weights[feature] ** 2 for feature in features)
        for group, features in FEATURE_GROUPS.items()
    }
    total = sum(group_energy.values())

    for group, budget in budgets.items():
        assert np.isclose(group_energy[group] / total, budget)


def test_temporal_targets_blend_only_adjacent_player_seasons() -> None:
    clean = np.array([[0.0], [2.0], [10.0]])
    targets = temporal_targets(
        clean,
        pd.Series([1, 1, 2]),
        pd.Series(["2020-21", "2021-22", "2020-21"]),
        .5,
    )

    assert np.allclose(targets[:, 0], [1.0, 1.0, 10.0])
