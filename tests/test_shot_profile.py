"""Tests for future shot-style embedding features."""

from __future__ import annotations

import pandas as pd
import pytest

from ball_ai.analytics.shot_profile import (
    build_broad_v2_shot_profiles,
    build_shot_profiles,
    engineer_shot_events,
)


def _shots() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (1, 1, 2, "Driving Dunk Shot", "Restricted Area"),
            (1, 0, 8, "Driving Floating Jump Shot", "In The Paint (Non-RA)"),
            (1, 1, 16, "Pullup Jump Shot", "Mid-Range"),
            (1, 0, 25, "Step Back Jump Shot", "Above the Break 3"),
            (2, 1, 3, "Layup Shot", "Restricted Area"),
            (2, 1, 3, "Layup Shot", "Restricted Area"),
        ],
        columns=[
            "player_id",
            "shot_made",
            "shot_distance",
            "action_type",
            "shot_zone_basic",
        ],
    )


def test_shot_events_preserve_specific_action_types() -> None:
    events = engineer_shot_events(_shots())
    assert events["action_bucket"].tolist()[:4] == [
        "dunk", "floater", "pull_up", "step_back"
    ]


def test_shot_profile_contains_mix_and_expected_shot_model() -> None:
    profile = build_shot_profiles(_shots(), prior_attempts=5).set_index("player_id")

    assert profile.loc[1, "rim_frequency"] == pytest.approx(0.25)
    assert profile.loc[1, "floater_frequency"] == pytest.approx(0.25)
    assert profile.loc[1, "shot_makes"] == 2
    assert profile.loc[1, "three_point_attempts"] == 1
    assert 0 <= profile.loc[1, "shot_difficulty_index"] <= 1
    assert profile.loc[2, "shot_making_above_expected"] > 0


def test_shot_profile_requires_explicit_schema() -> None:
    with pytest.raises(ValueError, match="Missing shot-event columns"):
        build_shot_profiles(pd.DataFrame({"player_id": [1]}))


def test_broad_v2_profile_uses_stable_distance_and_location_buckets() -> None:
    shots = _shots().assign(
        shot_type=["2PT", "2PT", "2PT", "3PT", "2PT", "2PT"],
        shot_zone_area=["Center(C)", "Center(C)", "Left Side(L)", "Right Side(R)", "Center(C)", "Center(C)"],
        loc_x=[0, 20, -120, 220, 0, 0],
    )
    profile = build_broad_v2_shot_profiles(shots).set_index("player_id")

    assert profile.loc[1, "distance_0_3_frequency"] == pytest.approx(0.25)
    assert profile.loc[1, "distance_8_15_frequency"] == pytest.approx(0.25)
    assert profile.loc[1, "distance_16_22_frequency"] == pytest.approx(0.25)
    assert profile.loc[1, "distance_23_27_frequency"] == pytest.approx(0.25)
    assert profile.loc[1, "corner_three_frequency"] == pytest.approx(0.25)
    assert profile.loc[1, "left_side_frequency"] == pytest.approx(0.25)
    assert profile.loc[1, "right_side_frequency"] == pytest.approx(0.25)
