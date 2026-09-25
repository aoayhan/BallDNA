"""Tests for player/roster embeddings and counterfactual reporting."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ball_ai.ai.grounding import build_roster_simulation_evidence
from ball_ai.ai.report_generator import generate_roster_simulation_report
from ball_ai.analytics.roster_construction import (
    MODEL_FEATURES,
    PLAYER_TRAITS,
    allocate_reference_rotation,
    assess_embedding_support,
    anchor_prediction_to_reference,
    assemble_simulated_roster,
    build_player_vectors,
    embedding_changes,
    evaluate_roster_models,
    evaluate_win_calibrators,
    fit_talent_ceiling_model,
    fit_win_calibrator,
    primary_team_player_pool,
    profile_scores,
    predict_talent_ceiling,
    roster_embedding,
    roster_talent_summary,
)


def _player_seasons() -> pd.DataFrame:
    rows = []
    for player_id, multiplier, position in (
        (1, 1.3, "G"), (2, 1.0, "F"), (3, .7, "C"),
        (6, .9, "F"), (7, .8, "C"),
    ):
        games = 60
        minutes = 25
        rows.append(
            {
                "season": "2024-25", "player_id": player_id,
                "player_name": f"Player {player_id}", "games_played": games,
                "minutes_per_game": minutes,
                "points_total": 900 * multiplier,
                "field_goal_attempts_total": 700 * multiplier,
                "field_goals_made_total": 350 * multiplier,
                "three_points_made_total": 100 * multiplier,
                "three_point_attempts_total": 280 * multiplier,
                "free_throw_attempts_total": 200 * multiplier,
                "assists_total": 250 * multiplier,
                "turnovers_total": 100,
                "offensive_rebounds_total": 80 / multiplier,
                "defensive_rebounds_total": 220 / multiplier,
                "steals_total": 65 * multiplier,
                "blocks_total": 45 / multiplier,
                "two_point_percentage": .54,
                "effective_field_goal_percentage": .55,
                "three_point_percentage": .36,
                "position": position,
                "points_per_game": 15 * multiplier,
                "rebounds_per_game": 6 / multiplier,
                "assists_per_game": 4 * multiplier,
            }
        )
    return pd.DataFrame(rows)


def test_player_vectors_and_minutes_weighted_roster_embedding() -> None:
    players = build_player_vectors(_player_seasons())
    assert all(f"trait__{trait}" in players for trait in PLAYER_TRAITS)
    assert players.filter(like="trait__").min().min() >= 0
    assert players.filter(like="trait__").max().max() <= 1
    assert players["role_strength"].between(0, 1).all()

    players["team"] = "AAA"
    players["team_minutes"] = [1800, 1200, 600, 900, 750]
    embedding = roster_embedding(players)
    assert set(MODEL_FEATURES).issubset(embedding)
    assert embedding["effective_rotation_size"] > 1
    assert 0 <= embedding["position_entropy"] <= 1


def test_external_impact_replaces_hand_weighted_quality_without_changing_traits() -> None:
    seasons = _player_seasons()
    impact = pd.DataFrame(
        {
            "season": ["2024-25", "2024-25"],
            "player_id": [1, 2],
            "dpm": [4.0, -2.0],
            "o_dpm": [3.0, -1.5],
            "d_dpm": [1.0, -0.5],
        }
    )

    players = build_player_vectors(seasons, impact_ratings=impact).set_index("player_id")

    assert players.loc[1, "impact_source"] == "DARKO DPM"
    assert players.loc[1, "quality_proxy"] > players.loc[2, "quality_proxy"]
    assert players.loc[3, "impact_source"] == "box-score fallback"


def test_rotation_scaling_preserves_every_positive_role() -> None:
    pool = build_player_vectors(_player_seasons())
    extra = pd.concat([pool.iloc[[0]].assign(player_id=20 + i) for i in range(6)])
    pool = pd.concat([pool, extra], ignore_index=True)

    roster = assemble_simulated_roster(pool, pool["player_id"].astype(int))

    assert np.isclose(roster["normalized_minutes"].sum(), 240.0)
    assert roster["normalized_minutes"].gt(0).all()


def test_primary_team_pool_keeps_only_a_traded_players_latest_stint() -> None:
    stints = pd.DataFrame(
        [
            {"season": "2025-26", "player_id": 1, "player_name": "Traded Player", "team": "OKC", "last_game_date": "2026-02-03", "team_minutes": 290},
            {"season": "2025-26", "player_id": 1, "player_name": "Traded Player", "team": "MIL", "last_game_date": "2026-04-12", "team_minutes": 797},
        ]
    )

    current = primary_team_player_pool(stints, "2025-26")

    assert current[["player_id", "team"]].to_dict("records") == [
        {"player_id": 1, "team": "MIL"}
    ]


def test_roster_assembly_normalizes_to_240_and_explains_changes() -> None:
    pool = build_player_vectors(_player_seasons())
    pool["team"] = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    roster = assemble_simulated_roster(
        pool, [1, 2, 3, 6, 7], {1: 36, 2: 30, 3: 24, 6: 20, 7: 10}
    )
    assert roster["normalized_minutes"].sum() == 240
    assert roster["normalized_minutes"].max() <= 48

    baseline = roster_embedding(roster, minute_column="normalized_minutes")
    changed = assemble_simulated_roster(
        pool, [1, 2, 3, 6, 7], {1: 40, 2: 32, 3: 20, 6: 16, 7: 8}
    )
    simulated = roster_embedding(changed, minute_column="normalized_minutes")
    changes = embedding_changes(baseline, simulated)
    assert len(changes) == len(PLAYER_TRAITS)
    assert changes.iloc[0]["absolute_change"] >= changes.iloc[-1]["absolute_change"]


def test_reference_anchor_preserves_observed_baseline_and_applies_model_delta() -> None:
    calibrator = LinearRegression().fit(
        pd.DataFrame({"net_rating": [-5.0, 0.0, 5.0]}), [28.0, 41.0, 54.0]
    )
    adjusted = anchor_prediction_to_reference(
        {"net_rating": 4.0, "expected_wins": 51.0},
        {"net_rating": 2.5, "expected_wins": 47.0},
        observed_net_rating=8.0,
        observed_wins=55,
        observed_games=82,
        win_calibrator=calibrator,
        net_rating_error=3.0,
    )
    assert adjusted["net_rating"] == 9.5
    assert adjusted["expected_wins"] == 59.0
    assert adjusted["wins_low"] < adjusted["expected_wins"] < adjusted["wins_high"]


def test_added_guard_displaces_guard_minutes_without_diluting_frontcourt() -> None:
    reference = build_player_vectors(_player_seasons())
    reference["team"] = "AAA"
    reference["team_minutes"] = 48.0
    low_value_guard = reference.iloc[[0]].copy()
    low_value_guard["player_id"] = 5
    low_value_guard["player_name"] = "Low-value Guard"
    low_value_guard["team_minutes"] = 10.0
    low_value_guard["minutes_per_game"] = 10.0
    low_value_guard["quality_proxy"] = 0.10
    low_value_guard["rotation_value"] = 0.10
    reference = pd.concat([reference, low_value_guard], ignore_index=True)
    added = reference.iloc[[0]].copy()
    added["player_id"] = 4
    added["player_name"] = "Added Guard"
    added["team"] = "BBB"
    added["position"] = "G-F"
    added["minutes_per_game"] = 25.0
    added["quality_proxy"] = 0.95
    added["rotation_value"] = 0.95
    pool = pd.concat([reference, added], ignore_index=True)

    baseline_rotation = allocate_reference_rotation(
        reference, pool, [1, 2, 3, 5, 6, 7]
    )
    simulated = allocate_reference_rotation(reference, pool, [1, 2, 3, 4, 5, 6, 7])
    minute_map = simulated.set_index("player_id")["normalized_minutes"]
    assert np.isclose(minute_map.sum(), 240.0)
    assert np.isclose(minute_map.loc[4], 25.0)
    assert np.isclose(minute_map.loc[5], 0.0)
    assert minute_map.loc[1] <= 48.0
    assert np.isclose(minute_map.loc[2], minute_map.loc[6])
    assert np.isclose(minute_map.loc[3], minute_map.loc[7])
    baseline_talent = roster_embedding(
        baseline_rotation, minute_column="normalized_minutes"
    )
    simulated_talent = roster_embedding(
        simulated, minute_column="normalized_minutes"
    )
    assert simulated_talent["rotation_quality"] >= baseline_talent["rotation_quality"]
    assert simulated_talent["top_three_quality"] > baseline_talent["top_three_quality"]

    baseline_embedding = roster_embedding(reference, minute_column="team_minutes")
    simulated_embedding = roster_embedding(simulated, minute_column="normalized_minutes")
    baseline_profiles = profile_scores(
        baseline_embedding, reference, minute_column="team_minutes"
    ).set_index("profile")
    simulated_profiles = profile_scores(
        simulated_embedding, simulated, minute_column="normalized_minutes"
    ).set_index("profile")
    assert np.isclose(
        baseline_profiles.loc["Frontcourt physicality", "score"],
        simulated_profiles.loc["Frontcourt physicality", "score"],
    )


def test_frontcourt_profile_is_calibrated_against_frontcourts() -> None:
    pool = build_player_vectors(_player_seasons())
    pool["normalized_minutes"] = 48.0
    roster = pool.loc[pool["position"].isin(["F", "C"])].copy()
    embedding = roster_embedding(roster, minute_column="normalized_minutes")

    raw = profile_scores(
        embedding, roster, minute_column="normalized_minutes"
    ).set_index("profile")
    calibrated = profile_scores(
        embedding,
        roster,
        minute_column="normalized_minutes",
        comparison_pool=pool,
    ).set_index("profile")

    assert calibrated.loc["Interior defensive tools", "score"] < raw.loc[
        "Interior defensive tools", "score"
    ]


def test_low_impact_addition_cannot_claim_unrelated_position_minutes() -> None:
    reference = build_player_vectors(_player_seasons())
    reference["team"] = "AAA"
    reference["team_minutes"] = 48.0
    addition = reference.iloc[[1]].copy()
    addition["player_id"] = 9
    addition["player_name"] = "Replacement-level Forward"
    addition["team"] = "BBB"
    addition["minutes_per_game"] = 30.0
    addition["quality_proxy"] = 0.05
    addition["rotation_value"] = 0.05
    reference["rotation_value"] = 0.60
    pool = pd.concat([reference, addition], ignore_index=True)

    simulated = allocate_reference_rotation(reference, pool, [1, 2, 3, 6, 7, 9])
    minute_map = simulated.set_index("player_id")["normalized_minutes"]
    assert np.isclose(minute_map.sum(), 240.0)
    assert np.isclose(minute_map.loc[9], 0.0)
    assert np.isclose(minute_map.loc[1], 48.0)
    assert np.isclose(minute_map.loc[3], 48.0)
    audit = simulated.attrs["allocation_audit"][0]
    assert audit["assigned_minutes"] == 0
    assert {item["player_name"] for item in audit["blocked_by"]} >= {
        "Player 2", "Player 6"
    }

    forced = allocate_reference_rotation(
        reference,
        pool,
        [1, 2, 3, 6, 7, 9],
        forced_player_ids=[9],
    )
    forced_minutes = forced.set_index("player_id")["normalized_minutes"]
    assert np.isclose(forced_minutes.sum(), 240.0)
    assert np.isclose(forced_minutes.loc[9], 30.0)
    assert np.isclose(
        forced_minutes.loc[[2, 6]].sum(),
        minute_map.loc[[2, 6]].sum() - 30.0,
    )
    forced_audit = forced.attrs["allocation_audit"][0]
    assert forced_audit["forced"] is True
    assert forced_audit["displacements"][0]["position"] == "F"
    assert np.isclose(forced_audit["displacements"][0]["minutes"], 30.0)


def test_rotation_allocation_is_independent_of_addition_order() -> None:
    reference = build_player_vectors(_player_seasons())
    reference["team"] = "AAA"
    reference["team_minutes"] = 48.0
    additions = []
    for player_id, impact in ((8, 0.75), (9, 0.95)):
        addition = reference.iloc[[0]].copy()
        addition["player_id"] = player_id
        addition["player_name"] = f"Added Guard {player_id}"
        addition["team"] = "BBB"
        addition["minutes_per_game"] = 30.0
        addition["quality_proxy"] = impact
        addition["rotation_value"] = impact
        additions.append(addition)
    reference["quality_proxy"] = 0.55
    reference["rotation_value"] = 0.55
    pool = pd.concat([reference, *additions], ignore_index=True)

    first = allocate_reference_rotation(reference, pool, [1, 2, 3, 6, 7, 8, 9])
    second = allocate_reference_rotation(reference, pool, [1, 2, 3, 6, 7, 9, 8])
    first_minutes = first.set_index("player_id")["normalized_minutes"].sort_index()
    second_minutes = second.set_index("player_id")["normalized_minutes"].sort_index()
    pd.testing.assert_series_equal(first_minutes, second_minutes)


def test_allocation_audit_is_strict_json_when_impact_rating_is_missing() -> None:
    reference = build_player_vectors(_player_seasons())
    reference["team"] = "AAA"
    reference["team_minutes"] = 48.0
    addition = reference.iloc[[0]].copy()
    addition["player_id"] = 99
    addition["player_name"] = "Fallback Addition"
    addition["team"] = "BBB"
    pool = pd.concat([reference, addition], ignore_index=True)

    simulated = allocate_reference_rotation(
        reference,
        pool,
        [*reference["player_id"].astype(int), 99],
    )

    json.dumps(simulated.attrs["allocation_audit"], allow_nan=False)


def test_bounded_win_calibration_has_diminishing_returns_at_elite_ratings() -> None:
    ratings = np.linspace(-12, 12, 90)
    wins = np.clip(np.rint(41 + 2.25 * ratings), 10, 72).astype(int)
    history = pd.DataFrame(
        {
            "season": np.repeat(["2022-23", "2023-24", "2024-25"], 30),
            "net_rating": ratings,
            "wins": wins,
            "games": 82,
        }
    )
    calibrator = fit_win_calibrator(history)
    predictions = calibrator.predict(
        pd.DataFrame({"net_rating": [0.0, 2.0, 10.0, 12.0]})
    )
    assert predictions.min() >= 0
    assert predictions.max() <= 82
    assert predictions[3] - predictions[2] < predictions[1] - predictions[0]
    scores = evaluate_win_calibrators(history)
    assert set(scores) == {"bounded_logistic_mae", "linear_mae"}


def test_model_comparison_and_rule_based_summary() -> None:
    rng = np.random.default_rng(42)
    rows = []
    for season_index, season in enumerate(("2021-22", "2022-23", "2023-24")):
        for team_id in range(8):
            row = {feature: float(rng.normal()) for feature in MODEL_FEATURES}
            row.update(
                {
                    "season": season,
                    "team_id": team_id,
                    "team": f"Team {team_id}",
                    "team_abbreviation": f"T{team_id}",
                    "games": 82,
                    "wins": 30 + team_id,
                    "league_rank": 8-team_id,
                }
            )
            row["net_rating"] = (
                3 * row["roster__scoring_efficiency"]
                + 2 * row["roster__playmaking"]
                + rng.normal(scale=.2)
            )
            rows.append(row)
    evaluation = evaluate_roster_models(pd.DataFrame(rows))
    assert set(evaluation["model"]) >= {"Elastic net", "Ridge regression", "Season-mean baseline"}
    assert evaluation["mae"].notna().all()

    baseline = {feature: 0.45 for feature in MODEL_FEATURES}
    simulated = dict(baseline)
    simulated["roster__scoring_efficiency"] = 0.65
    simulated["roster__playmaking"] = 0.60
    changes = embedding_changes(baseline, simulated)
    profiles = profile_scores(simulated)
    roster = pd.DataFrame(
        [{
            "player_id": 1, "player_name": "Player 1", "team": "AAA",
            "games_played": 60, "minutes_per_game": 30.0,
            "normalized_minutes": 240.0, "model_rotation_share": 1.0,
            "role_strength": .8, "data_reliability": .9,
        }]
    )
    neighbours = pd.DataFrame(
        [{
            "team": "Example Team", "season": "2023-24", "wins": 50, "games": 82,
            "net_rating": 4.2, "similarity": .8,
        }]
    )
    prediction = {
        "net_rating": 2.0, "wins_low": 40.0, "wins_high": 50.0,
    }
    packet = build_roster_simulation_evidence(
        season="2024-25",
        baseline_team={"team": "Baseline", "wins": 40, "games": 82},
        baseline_prediction={"net_rating": 0.0},
        simulated_prediction=prediction,
        roster=roster,
        added_players=["Player 1"],
        removed_players=[],
        changes=changes,
        profiles=profiles,
        neighbours=neighbours,
        model_evaluation={
            "selected_model": "Elastic net", "selected_model_mae": 1.8,
            "training_rows": 150, "seasons": ["2020-21"], "limitations": [],
        },
    )
    report = generate_roster_simulation_report(packet)
    assert report.mode == "rule-based"
    assert "## Simulation read" in report.text
    assert "[V1]" in report.text
    assert "[C1]" in report.text


def test_out_of_distribution_roster_withholds_quality_estimate() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for index in range(24):
        row = {feature: float(rng.normal(0.5, 0.025)) for feature in MODEL_FEATURES}
        row.update(
            {
                "season": "2024-25",
                "team_id": index,
                "team": f"Team {index}",
                "team_abbreviation": f"T{index}",
                "net_rating": 20
                * (
                    row["rotation_quality"]
                    + row["top_three_quality"]
                    - 1.0
                ),
            }
        )
        rows.append(row)
    history = pd.DataFrame(rows)
    extreme = {feature: 0.95 for feature in MODEL_FEATURES}
    support = assess_embedding_support(extreme, history)
    talent = roster_talent_summary(extreme, history)
    assert support["is_supported"] is False
    assert support["distance_ratio"] > 1
    assert talent["historical_percentile"] == 100
    ceiling_model = fit_talent_ceiling_model(history)
    calibrator = LinearRegression().fit(
        pd.DataFrame({"net_rating": [-10.0, 0.0, 10.0]}), [20.0, 41.0, 62.0]
    )
    ordinary = {feature: 0.5 for feature in MODEL_FEATURES}
    assert predict_talent_ceiling(
        extreme, ceiling_model, calibrator
    )["net_rating"] > predict_talent_ceiling(
        ordinary, ceiling_model, calibrator
    )["net_rating"]

    roster = pd.DataFrame(
        [{
            "player_id": 1, "player_name": "All Star", "team": "AAA",
            "games_played": 70, "minutes_per_game": 35.0,
            "model_rotation_share": 1.0, "role_strength": .99,
            "data_reliability": .95,
        }]
    )
    profiles = profile_scores(extreme)
    packet = build_roster_simulation_evidence(
        season="2024-25",
        baseline_team={"team": "Baseline", "wins": 40, "games": 82},
        baseline_prediction={"net_rating": 0.0},
        simulated_prediction={"net_rating": -9.0, "wins_low": 20.0, "wins_high": 30.0},
        roster=roster,
        added_players=["All Star"],
        removed_players=[],
        changes=embedding_changes({feature: .5 for feature in MODEL_FEATURES}, extreme),
        profiles=profiles,
        neighbours=pd.DataFrame(
            [{
                "team": "Example", "season": "2023-24", "wins": 50,
                "games": 82, "net_rating": 4.0, "similarity": .2,
            }]
        ),
        model_evaluation={"selected_model_mae": 1.8, "limitations": []},
        support=support,
        talent=talent,
    )
    assert packet["quality_estimates"] == []
    report = generate_roster_simulation_report(packet)
    assert "does not receive a validated expected Net Rating or win change" in report.text
    assert "-9.0" not in report.text
    assert "[S1]" in report.text
    assert "[T1]" in report.text
