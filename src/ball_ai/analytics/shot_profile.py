"""Shot-event feature engineering for a future style-aware player embedding.

This module deliberately accepts normalized events rather than fetching data.
That keeps licensing/provenance decisions in the ingestion layer and makes the
analytics independently testable with synthetic fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_SHOT_COLUMNS = {
    "player_id",
    "shot_made",
    "shot_distance",
    "action_type",
    "shot_zone_basic",
}

SHOT_PROFILE_FEATURES = [
    "rim_frequency",
    "paint_frequency",
    "midrange_frequency",
    "three_point_frequency",
    "dunk_frequency",
    "layup_frequency",
    "floater_frequency",
    "hook_frequency",
    "pull_up_frequency",
    "step_back_frequency",
    "average_shot_distance",
    "shot_difficulty_index",
    "shot_making_above_expected",
]

BROAD_V2_REQUIRED_SHOT_COLUMNS = {
    *REQUIRED_SHOT_COLUMNS,
    "shot_type",
    "shot_zone_area",
    "loc_x",
}

BROAD_V2_SHOT_FEATURES = [
    "distance_0_3_frequency",
    "distance_4_7_frequency",
    "distance_8_15_frequency",
    "distance_16_22_frequency",
    "distance_23_27_frequency",
    "distance_28_39_frequency",
    "distance_40_plus_frequency",
    "restricted_area_frequency",
    "non_restricted_paint_frequency",
    "corner_three_frequency",
    "above_break_three_frequency",
    "left_side_frequency",
    "center_channel_frequency",
    "right_side_frequency",
    "shot_distance_std",
    "lateral_location_std",
]


def _zone_bucket(zone: str, distance: float) -> str:
    value = str(zone).lower()
    if "restricted" in value or distance <= 4:
        return "rim"
    if "paint" in value:
        return "paint"
    if "mid-range" in value or "midrange" in value:
        return "midrange"
    if "3" in value or distance >= 23:
        return "three_point"
    return "other"


def _action_bucket(action: str) -> str:
    value = str(action).lower()
    # Specific moves must be checked before their more generic families.
    if "step back" in value or "step-back" in value:
        return "step_back"
    if "pullup" in value or "pull-up" in value or "pull up" in value:
        return "pull_up"
    if "float" in value:
        return "floater"
    if "hook" in value:
        return "hook"
    if "dunk" in value:
        return "dunk"
    if "layup" in value or "finger roll" in value:
        return "layup"
    return "other"


def engineer_shot_events(shots: pd.DataFrame) -> pd.DataFrame:
    """Classify normalized shots into transparent zone and action buckets."""

    missing = REQUIRED_SHOT_COLUMNS - set(shots.columns)
    if missing:
        raise ValueError(f"Missing shot-event columns: {sorted(missing)}")
    events = shots.copy()
    events["shot_made"] = pd.to_numeric(events["shot_made"], errors="coerce").fillna(0).clip(0, 1)
    events["shot_distance"] = pd.to_numeric(
        events["shot_distance"], errors="coerce"
    ).fillna(0).clip(lower=0)
    events["zone_bucket"] = [
        _zone_bucket(zone, distance)
        for zone, distance in zip(events["shot_zone_basic"], events["shot_distance"])
    ]
    events["action_bucket"] = events["action_type"].map(_action_bucket)
    return events


def build_shot_profiles(
    shots: pd.DataFrame,
    *,
    prior_attempts: float = 25.0,
) -> pd.DataFrame:
    """Aggregate shot mix and a smoothed expected-FG difficulty proxy.

    Expected FG% is estimated from league-wide zone/action buckets with Bayesian
    shrinkage toward the overall mean. ``shot_difficulty_index`` is one minus a
    player's expected FG%; ``shot_making_above_expected`` is actual minus expected.
    These are descriptive model outputs, not direct measures of defender pressure.
    """

    events = engineer_shot_events(shots)
    if events.empty:
        return pd.DataFrame(
            columns=[
                "player_id", "shot_attempts", "shot_makes",
                "three_point_attempts", "three_point_makes",
                *SHOT_PROFILE_FEATURES,
            ]
        )
    if prior_attempts < 0:
        raise ValueError("prior_attempts must be non-negative.")

    league_mean = float(events["shot_made"].mean())
    baselines = (
        events.groupby(["zone_bucket", "action_bucket"], dropna=False)["shot_made"]
        .agg(["sum", "count"])
        .reset_index()
    )
    baselines["expected_fg"] = (
        baselines["sum"] + prior_attempts * league_mean
    ) / (baselines["count"] + prior_attempts)
    events = events.merge(
        baselines[["zone_bucket", "action_bucket", "expected_fg"]],
        on=["zone_bucket", "action_bucket"],
        how="left",
        validate="many_to_one",
    )

    records = []
    for player_id, player_shots in events.groupby("player_id", sort=False):
        attempts = len(player_shots)
        zone_frequency = player_shots["zone_bucket"].value_counts(normalize=True)
        action_frequency = player_shots["action_bucket"].value_counts(normalize=True)
        actual_fg = float(player_shots["shot_made"].mean())
        expected_fg = float(player_shots["expected_fg"].mean())
        records.append(
            {
                "player_id": int(player_id),
                "shot_attempts": int(attempts),
                "shot_makes": int(player_shots["shot_made"].sum()),
                "three_point_attempts": int(
                    player_shots["zone_bucket"].eq("three_point").sum()
                ),
                "three_point_makes": int(
                    player_shots.loc[
                        player_shots["zone_bucket"].eq("three_point"), "shot_made"
                    ].sum()
                ),
                "rim_frequency": float(zone_frequency.get("rim", 0.0)),
                "paint_frequency": float(zone_frequency.get("paint", 0.0)),
                "midrange_frequency": float(zone_frequency.get("midrange", 0.0)),
                "three_point_frequency": float(zone_frequency.get("three_point", 0.0)),
                "dunk_frequency": float(action_frequency.get("dunk", 0.0)),
                "layup_frequency": float(action_frequency.get("layup", 0.0)),
                "floater_frequency": float(action_frequency.get("floater", 0.0)),
                "hook_frequency": float(action_frequency.get("hook", 0.0)),
                "pull_up_frequency": float(action_frequency.get("pull_up", 0.0)),
                "step_back_frequency": float(action_frequency.get("step_back", 0.0)),
                "average_shot_distance": float(player_shots["shot_distance"].mean()),
                "shot_difficulty_index": float(np.clip(1.0 - expected_fg, 0.0, 1.0)),
                "shot_making_above_expected": actual_fg - expected_fg,
            }
        )
    return pd.DataFrame(records)


def build_broad_v2_shot_profiles(shots: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stable location features available throughout the 1996+ archive."""

    missing = BROAD_V2_REQUIRED_SHOT_COLUMNS - set(shots.columns)
    if missing:
        raise ValueError(f"Missing broad-v2 shot columns: {sorted(missing)}")
    events = engineer_shot_events(shots)
    distance = events["shot_distance"]
    area = events["shot_zone_area"].astype("string").str.lower()
    zone = events["shot_zone_basic"].astype("string").str.lower()
    shot_type = events["shot_type"].astype("string").str.lower()
    loc_x = pd.to_numeric(events["loc_x"], errors="coerce")
    is_three = shot_type.str.contains("3pt", na=False) | zone.str.contains("3", na=False)
    is_corner = is_three & (
        zone.str.contains("corner", na=False)
        | area.isin(["left side(l)", "right side(r)"])
    )

    flags = {
        "distance_0_3_frequency": distance.le(3),
        "distance_4_7_frequency": distance.between(4, 7),
        "distance_8_15_frequency": distance.between(8, 15),
        "distance_16_22_frequency": distance.between(16, 22),
        "distance_23_27_frequency": distance.between(23, 27),
        "distance_28_39_frequency": distance.between(28, 39),
        "distance_40_plus_frequency": distance.ge(40),
        "restricted_area_frequency": zone.str.contains("restricted", na=False),
        "non_restricted_paint_frequency": zone.str.contains("paint", na=False)
        & ~zone.str.contains("restricted", na=False),
        "corner_three_frequency": is_corner,
        "above_break_three_frequency": is_three & ~is_corner & ~area.str.contains("back", na=False),
    }
    for name, values in flags.items():
        events[name] = values.astype(float)
    events["left_side_frequency"] = (loc_x < -80).where(loc_x.notna()).astype(float)
    events["center_channel_frequency"] = loc_x.abs().le(80).where(loc_x.notna()).astype(float)
    events["right_side_frequency"] = (loc_x > 80).where(loc_x.notna()).astype(float)
    events["loc_x"] = loc_x

    grouped = events.groupby("player_id", sort=False)
    output = grouped[[*flags, "left_side_frequency", "center_channel_frequency", "right_side_frequency"]].mean()
    output["shot_distance_std"] = grouped["shot_distance"].std(ddof=0)
    output["lateral_location_std"] = grouped["loc_x"].std(ddof=0).div(250)
    return output.reset_index()[["player_id", *BROAD_V2_SHOT_FEATURES]]
