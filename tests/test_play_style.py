"""Checks for player-style feature engineering and retrieval."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ball_ai.analytics.play_style import (
    BROAD_OFFENSIVE_FEATURES,
    MODERN_MOVE_FEATURES,
    OFFENSIVE_FEATURES,
    STYLE_FEATURE_SETS,
    build_action_profiles,
    build_defensive_matchup_profiles,
    build_shot_context_features,
    find_style_neighbors,
    fit_temporal_contrastive_artifact,
    historical_self_similarities,
    input_feature_comparison,
    shot_sample_reliability,
    split_feature_explanations,
    temporal_self_match,
    top_siamese_style_pairs,
    transform_temporal_contrastive,
)


def test_broad_history_excludes_era_limited_move_labels() -> None:
    assert not set(MODERN_MOVE_FEATURES) & set(BROAD_OFFENSIVE_FEATURES)
    assert set(MODERN_MOVE_FEATURES) <= set(OFFENSIVE_FEATURES)
    assert "three_point_frequency" not in BROAD_OFFENSIVE_FEATURES
    assert "three_point_attempt_rate" in BROAD_OFFENSIVE_FEATURES


def test_feature_comparison_can_hide_unavailable_era_features() -> None:
    features = pd.DataFrame({
        "player_id": [1, 2],
        "season": ["2001-02"] * 2,
        "rim_frequency": [.6, .5],
        "floater_frequency": [0.0, 0.0],
    })
    values = features[["rim_frequency", "floater_frequency"]]
    artifact = {
        "feature_names": values.columns.tolist(),
        "imputer": SimpleImputer().fit(values),
        "scaler": StandardScaler().fit(values),
        "feature_weights": np.ones(2),
        "training_end": "2001-02",
    }

    comparison = input_feature_comparison(
        features, artifact, 1, 2, "2001-02", feature_names=["rim_frequency"]
    )

    assert comparison["Feature"].tolist() == ["Rim shot frequency"]


def test_feature_explanation_lists_do_not_overlap() -> None:
    comparison = pd.DataFrame({
        "Feature": ["A", "B", "C", "D"],
        "Standardized gap": [4.0, 1.0, 3.0, 0.5],
        "Shared tendency strength": [4.0, 3.0, 2.0, 1.0],
    })

    shared, different = split_feature_explanations(comparison, 2)

    assert different["Feature"].tolist() == ["A", "C"]
    assert shared["Feature"].tolist() == ["B", "D"]
    assert set(shared["Feature"]).isdisjoint(different["Feature"])


def test_presence_aware_retrieval_rewards_shared_actions_not_shared_zeros() -> None:
    embeddings = pd.DataFrame({
        "player_id": [1, 2, 3],
        "player_name": ["Reference", "Shared absence", "Shared tendency"],
        "season": ["2025-26"] * 3,
        "position": ["C"] * 3,
        "dpm": [0.0] * 3,
        "o_dpm": [0.0] * 3,
        "d_dpm": [0.0] * 3,
        "eligible": [True] * 3,
        "lens": ["Offensive"] * 3,
        "method": ["Denoising autoencoder"] * 3,
        "embedding_00": [1.0, 0.999, 0.98],
        "embedding_01": [0.0, np.sqrt(1 - 0.999**2), np.sqrt(1 - 0.98**2)],
    })
    features = pd.DataFrame({
        "player_id": [1, 2, 3],
        "season": ["2025-26"] * 3,
        "dunk_frequency": [0.20, 0.02, 0.18],
        "step_back_frequency": [0.0, 0.0, 0.08],
    })
    training = features[["dunk_frequency", "step_back_frequency"]]
    imputer = SimpleImputer(strategy="median").fit(training)
    artifact = {
        "feature_names": training.columns.tolist(),
        "imputer": imputer,
        "scaler": StandardScaler().fit(imputer.transform(training)),
        "feature_weights": np.ones(2),
    }

    result = find_style_neighbors(
        embeddings, 1, lens="Offensive", method="Denoising autoencoder",
        season="2025-26", top_n=2, features=features, artifact=artifact,
        presence_weight=0.4,
    )
    explanation = input_feature_comparison(
        features, artifact, 1, 2, "2025-26"
    ).set_index("Feature")

    assert result.iloc[0]["player_name"] == "Shared tendency"
    assert explanation.loc["Step-back frequency", "Shared tendency strength"] == 0
    assert explanation.loc["Step-back frequency", "Weak shared-absence evidence"] > 0
    assert explanation.loc["Dunk frequency", "Shared tendency strength"] > 0


def test_unique_player_retrieval_keeps_only_best_season() -> None:
    embeddings = pd.DataFrame({
        "player_id": [1, 2, 2, 3],
        "player_name": ["Reference", "Repeat", "Repeat", "Other"],
        "season": ["2025-26", "2024-25", "2023-24", "2024-25"],
        "position": ["G"] * 4,
        "dpm": [0.0] * 4,
        "o_dpm": [0.0] * 4,
        "d_dpm": [0.0] * 4,
        "eligible": [True] * 4,
        "lens": ["Offensive"] * 4,
        "method": ["Denoising autoencoder"] * 4,
        "embedding_00": [1.0, .99, .98, .97],
        "embedding_01": [0.0, np.sqrt(1 - .99**2), np.sqrt(1 - .98**2), np.sqrt(1 - .97**2)],
    })

    result = find_style_neighbors(
        embeddings, 1, lens="Offensive", method="Denoising autoencoder",
        season="2025-26", top_n=3, unique_players=True,
    )

    assert result["player_name"].tolist() == ["Repeat", "Other"]
    assert result.iloc[0]["season"] == "2024-25"


def test_precomputed_siamese_component_changes_retrieval_order() -> None:
    embeddings = pd.DataFrame({
        "player_id": [1, 2, 3], "player_name": ["Reference", "Base", "Siamese"],
        "season": ["2025-26"] * 3, "position": ["G"] * 3,
        "dpm": [0.0] * 3, "o_dpm": [0.0] * 3, "d_dpm": [0.0] * 3,
        "eligible": [True] * 3, "lens": ["Offensive"] * 3,
        "method": ["Denoising autoencoder"] * 3,
        "embedding_00": [1.0, 1.0, 0.0], "embedding_01": [0.0, 0.0, 1.0],
    })
    siamese = pd.DataFrame({
        "player_id": [1, 2, 3], "season": ["2025-26"] * 3,
        "siamese_embedding_00": [1.0, 0.0, 1.0],
        "siamese_embedding_01": [0.0, 1.0, 0.0],
    })

    result = find_style_neighbors(
        embeddings, 1, lens="Offensive", method="Denoising autoencoder",
        season="2025-26", siamese_embeddings=siamese, siamese_weight=.6,
    )

    assert result.iloc[0]["player_name"] == "Siamese"


def test_detailed_siamese_requires_coverage_for_both_seasons() -> None:
    embeddings = pd.DataFrame({
        "player_id": [1, 2, 3], "player_name": ["Reference", "Stable", "Detailed"],
        "season": ["2025-26", "2024-25", "2024-25"], "position": ["G"] * 3,
        "dpm": [0.0] * 3, "o_dpm": [0.0] * 3, "d_dpm": [0.0] * 3,
        "eligible": [True] * 3, "lens": ["Offensive"] * 3,
        "method": ["Denoising autoencoder"] * 3,
        "embedding_00": [1.0] * 3,
    })
    siamese = pd.DataFrame({
        "player_id": [1, 2, 3], "season": embeddings["season"],
        "siamese_embedding_00": [1.0, 1.0, 0.0],
        "siamese_embedding_01": [0.0, 0.0, 1.0],
        "detailed_siamese_embedding_00": [1.0, 0.0, 1.0],
        "detailed_siamese_embedding_01": [0.0, 1.0, 0.0],
    })

    modern = find_style_neighbors(
        embeddings, 1, lens="Offensive", method="Denoising autoencoder",
        season="2025-26", siamese_embeddings=siamese, siamese_weight=1.0,
        siamese_detail_weight=.75,
    )
    early_embeddings = embeddings.assign(season="2006-07")
    early_siamese = siamese.assign(season="2006-07")
    early = find_style_neighbors(
        early_embeddings, 1, lens="Offensive", method="Denoising autoencoder",
        season="2006-07", siamese_embeddings=early_siamese, siamese_weight=1.0,
    )

    assert modern.iloc[0]["player_name"] == "Detailed"
    assert early.iloc[0]["player_name"] == "Stable"


def test_siamese_leaderboard_excludes_self_and_mirrored_pairs() -> None:
    vectors = pd.DataFrame({
        "player_id": [1, 1, 2, 3],
        "season": ["2024-25", "2023-24", "2024-25", "2024-25"],
        "siamese_embedding_00": [1.0, 1.0, 1.0, 0.0],
        "siamese_embedding_01": [0.0, 0.0, 0.0, 1.0],
        "detailed_siamese_embedding_00": [1.0, 1.0, 1.0, 0.0],
        "detailed_siamese_embedding_01": [0.0, 0.0, 0.0, 1.0],
    })
    metadata = pd.DataFrame({
        "player_id": [1, 1, 2, 3],
        "season": vectors["season"],
        "player_name": ["One", "One", "Two", "Three"],
    })

    leaderboard = top_siamese_style_pairs(vectors, metadata, top_n=10)

    assert not (leaderboard["player_a_id"] == leaderboard["player_b_id"]).any()
    assert len(leaderboard) == 5
    assert leaderboard.iloc[0]["similarity_score"] == 1.0


def test_direct_style_comparison_targets_requested_player_season() -> None:
    embeddings = pd.DataFrame({
        "player_id": [1, 2, 2, 3],
        "player_name": ["Reference", "Target", "Target", "Closer other"],
        "season": ["2025-26", "2024-25", "2023-24", "2024-25"],
        "position": ["G"] * 4,
        "dpm": [1.0, 2.0, 1.5, 3.0],
        "o_dpm": [1.0, 2.0, 1.5, 3.0],
        "d_dpm": [0.0] * 4,
        "eligible": [True] * 4,
        "lens": ["Offensive"] * 4,
        "method": ["Denoising autoencoder"] * 4,
        "embedding_00": [1.0, .8, .7, .99],
        "embedding_01": [0.0, .6, np.sqrt(1 - .7**2), np.sqrt(1 - .99**2)],
    })

    result = find_style_neighbors(
        embeddings, 1, lens="Offensive", method="Denoising autoencoder",
        season="2025-26", candidate_player_id=2,
        candidate_season_start="2024-25", candidate_season_end="2024-25",
        top_n=1,
    )

    assert result.iloc[0]["player_name"] == "Target"
    assert result.iloc[0]["season"] == "2024-25"
    assert result.iloc[0]["o_dpm_difference"] == 1.0


def test_temporal_encoder_uses_behavior_only_and_returns_normalized_vectors() -> None:
    rows = []
    for player_id, offset in ((1, 0.0), (2, 1.0)):
        for season_index, season in enumerate(("2022-23", "2023-24")):
            row = {
                "player_id": player_id,
                "season": season,
                "offensive_eligible": True,
            }
            row.update({
                feature: offset + season_index * 0.01 + feature_index * 0.001
                for feature_index, feature in enumerate(STYLE_FEATURE_SETS["Broad history"]["Offensive"])
            })
            rows.append(row)
    features = pd.DataFrame(rows)

    artifact = fit_temporal_contrastive_artifact(
        features, training_end="2023-24", latent_dimensions=2, max_iter=5
    )
    vectors = transform_temporal_contrastive(artifact, features)

    assert "player_id" not in artifact["feature_names"]
    assert vectors.shape == (4, 2)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)


def test_style_spaces_do_not_use_impact_as_an_input() -> None:
    forbidden = {
        "dpm", "o_dpm", "d_dpm", "plus_minus", "wins",
        "player_id", "player_name", "team", "position",
        "height_inches", "weight_lbs", "matchup_opponent_height",
    }
    assert not forbidden.intersection(
        feature
        for lenses in STYLE_FEATURE_SETS.values()
        for values in lenses.values()
        for feature in values
    )


def test_shot_context_keeps_ineligible_splits_missing() -> None:
    shots = pd.DataFrame({
        "PLAYER_ID": [1, 1, 1, 1],
        "SHOT_MADE_FLAG": [1, 0, 1, 0],
        "SHOT_DISTANCE": [2, 25, 4, 18],
        "ACTION_TYPE": ["Layup", "Jump Shot", "Dunk", "Pullup Jump Shot"],
        "SHOT_ZONE_BASIC": ["Restricted Area", "Above the Break 3", "Restricted Area", "Mid-Range"],
        "TEAM_ID": [1610612760] * 4,
        "PERIOD": [1, 2, 4, 4],
        "HTM": ["OKC", "BOS", "OKC", "BOS"],
        "_season": [2024] * 4,
        "_season_type": ["rg"] * 4,
    })
    result = build_shot_context_features(shots).iloc[0]
    assert not result["home_away_context_eligible"]
    assert not result["quarter_context_eligible"]
    assert np.isnan(result["home_away_zone_shift"])


def test_defensive_profiles_use_biographies_for_missing_positions() -> None:
    matchups = pd.DataFrame({
        "matchups_person_id": [9, 9],
        "person_id": [1, 2],
        "position": [None, None],
        "game_id": [10, 11],
        "partial_possessions": [60.0, 40.0],
        "matchup_field_goals_attempted": [10, 8],
        "matchup_three_pointers_attempted": [5, 1],
        "matchup_turnovers": [2, 1],
        "matchup_blocks": [1, 0],
        "shooting_fouls": [1, 1],
        "_season": [2024, 2024],
        "_season_type": ["rg", "rg"],
    })
    bios = pd.DataFrame({
        "player_id": [1, 2],
        "guard": [1, 0],
        "forward": [0, 0],
        "center": [0, 1],
        "heightInches": [74, 83],
    })
    result = build_defensive_matchup_profiles(matchups, bios).iloc[0]
    assert result["matchup_guard_share"] == 0.6
    assert result["matchup_center_share"] == 0.4
    assert result["matchup_opponent_height"] == 77.6


def test_action_profiles_use_tagged_actions_and_possession_resets() -> None:
    actions = pd.DataFrame({
        "actionType": ["2pt", "2pt", "turnover", "2pt", "rebound", "2pt"],
        "subType": ["Layup", "Jump Shot", "bad pass", "DUNK", "offensive", "Layup"],
        "qualifiers": [None, "fastbreak", None, "fromturnover", None, "2ndchance"],
        "personId": [2, 1, 2, 1, 1, 1],
        "assistPersonId": [np.nan, 3, np.nan, np.nan, np.nan, np.nan],
        "possession": [20, 10, 20, 10, 10, 10],
        "orderNumber": [1, 2, 3, 4, 5, 6],
        "clock": ["PT11M50.00S", "PT11M40.00S", "PT11M30.00S", "PT11M26.00S", "PT11M24.00S", "PT11M14.00S"],
        "period": [1] * 6,
        "gameId": [99] * 6,
        "shotResult": ["Made", "Made", None, "Made", None, "Made"],
        "_season": [2025] * 6,
        "_season_type": ["rg"] * 6,
    })
    result = build_action_profiles(actions).set_index("player_id").loc[1]
    assert result["fastbreak_shot_rate"] == 1 / 3
    assert result["turnover_created_shot_rate"] == 1 / 3
    assert result["second_chance_shot_rate"] == 1 / 3
    assert result["early_clock_shot_rate"] == 1 / 3
    assert result["late_clock_shot_rate"] == 1 / 3


def test_learned_retrieval_excludes_ineligible_candidates_and_keeps_impact_separate() -> None:
    frame = pd.DataFrame({
        "player_id": [1, 2, 3],
        "player_name": ["Reference", "Close", "Ineligible"],
        "season": ["2025-26"] * 3,
        "position": ["G"] * 3,
        "games_played": [70, 70, 3],
        "shot_attempts": [1000, 900, 20],
        "dpm": [5.0, 1.0, 6.0],
        "o_dpm": [4.0, 1.0, 5.0],
        "d_dpm": [1.0, 0.0, 1.0],
        "eligible": [True, True, False],
        "offensive_reliability": [1.0, 1.0, 0.1],
        "lens": ["Offensive"] * 3,
        "method": ["Denoising autoencoder"] * 3,
        "embedding_00": [1.0, 0.99, 1.0],
        "embedding_01": [0.0, 0.1, 0.0],
    })
    result = find_style_neighbors(
        frame,
        1,
        lens="Offensive",
        method="Denoising autoencoder",
        season="2025-26",
    )
    assert result["player_name"].tolist() == ["Close"]
    assert result.iloc[0]["dpm_difference"] == -4.0


def test_learned_retrieval_rejects_ineligible_reference() -> None:
    frame = pd.DataFrame({
        "player_id": [1, 2], "player_name": ["Reference", "Candidate"],
        "season": ["2025-26"] * 2, "position": ["G"] * 2,
        "dpm": [0.0] * 2, "o_dpm": [0.0] * 2, "d_dpm": [0.0] * 2,
        "eligible": [False, True], "lens": ["Offensive"] * 2,
        "method": ["Denoising autoencoder"] * 2,
        "embedding_00": [1.0, .9], "embedding_01": [0.0, .1],
    })

    with np.testing.assert_raises_regex(ValueError, "reference player-season"):
        find_style_neighbors(
            frame, 1, lens="Offensive", method="Denoising autoencoder",
            season="2025-26",
        )


def test_temporal_self_match_reports_rank_and_similarity() -> None:
    frame = pd.DataFrame({
        "player_id": [1, 1, 2],
        "season": ["2025-26", "2024-25", "2024-25"],
        "lens": ["Offensive"] * 3,
        "method": ["Denoising autoencoder"] * 3,
        "eligible": [True] * 3,
        "embedding_00": [1.0, np.cos(0.01), 0.0],
        "embedding_01": [0.0, np.sin(0.01), 1.0],
    })
    result = temporal_self_match(
        frame,
        1,
        lens="Offensive",
        method="Denoising autoencoder",
        query_season="2025-26",
        candidate_season="2024-25",
    )
    assert result["rank"] == 1
    assert result["candidate_count"] == 2
    assert result["similarity_score"] > 0.99


def test_comparable_player_filters_by_lens_specific_impact() -> None:
    frame = pd.DataFrame({
        "player_id": [1, 2, 3],
        "player_name": ["Reference", "Style only", "Comparable"],
        "season": ["2025-26"] * 3,
        "position": ["G"] * 3,
        "games_played": [70] * 3,
        "shot_attempts": [900] * 3,
        "dpm": [5.0, 0.0, 4.5],
        "o_dpm": [4.0, -1.0, 3.5],
        "d_dpm": [1.0, 1.0, 1.0],
        "eligible": [True] * 3,
        "offensive_reliability": [1.0] * 3,
        "lens": ["Offensive"] * 3,
        "method": ["Denoising autoencoder"] * 3,
        "embedding_00": [1.0, 0.999, 0.95],
        "embedding_01": [0.0, 0.045, 0.312],
    })
    result = find_style_neighbors(
        frame,
        1,
        lens="Offensive",
        method="Denoising autoencoder",
        season="2025-26",
        impact_tolerance=1.0,
    )
    assert result["player_name"].tolist() == ["Comparable"]


def test_shot_sample_reliability_uses_bootstrap_bands() -> None:
    assert shot_sample_reliability(100) == "Limited"
    assert shot_sample_reliability(399) == "Limited"
    assert shot_sample_reliability(400) == "Medium"
    assert shot_sample_reliability(800) == "High"
    assert shot_sample_reliability(None) == "Unavailable"


def test_comparable_player_explains_missing_reference_impact() -> None:
    frame = pd.DataFrame({
        "player_id": [1, 2], "player_name": ["Reference", "Candidate"],
        "season": ["2020-21"] * 2, "position": ["F", "G"],
        "dpm": [np.nan, np.nan], "o_dpm": [np.nan, np.nan],
        "d_dpm": [np.nan, np.nan], "eligible": [True, True],
        "lens": ["Offensive"] * 2, "method": ["Denoising autoencoder"] * 2,
        "embedding_00": [1.0, .9], "embedding_01": [0.0, .1],
    })
    with np.testing.assert_raises_regex(ValueError, "Use Style Twin instead"):
        find_style_neighbors(
            frame, 1, lens="Offensive", method="Denoising autoencoder",
            season="2020-21", impact_tolerance=1.5,
        )


def test_historical_retrieval_separates_prior_self_from_player_results() -> None:
    frame = pd.DataFrame({
        "player_id": [1, 1, 2, 3, 4],
        "player_name": ["Reference", "Reference", "Too old", "Too early", "Other player"],
        "season": ["2025-26", "2024-25", "2024-25", "2005-06", "2024-25"],
        "age": [27, 26, 35, 25, 26],
        "position": ["G"] * 5,
        "dpm": [5.0, 4.0, 2.0, 1.0, 3.0],
        "o_dpm": [4.0, 3.0, 1.0, 0.0, 2.0],
        "d_dpm": [1.0] * 5,
        "eligible": [True] * 5,
        "lens": ["Overall"] * 5,
        "method": ["Denoising autoencoder"] * 5,
        "embedding_00": [1.0, 0.99, 0.98, 0.97, 0.96],
        "embedding_01": [0.0, 0.1, 0.2, 0.25, 0.28],
    })
    result = find_style_neighbors(
        frame,
        1,
        lens="Overall",
        method="Denoising autoencoder",
        season="2025-26",
        candidate_season_start="2014-15",
        candidate_season_end="2024-25",
        minimum_age=24,
        maximum_age=30,
    )
    assert result[["player_name", "season"]].values.tolist() == [["Other player", "2024-25"]]
    history = historical_self_similarities(
        frame, 1, lens="Overall", method="Denoising autoencoder",
        season="2025-26",
    )
    assert history[["player_name", "season"]].values.tolist() == [["Reference", "2024-25"]]


def test_profile_depths_use_separate_candidate_spaces() -> None:
    frame = pd.DataFrame({
        "player_id": [1, 2, 3, 1, 2, 3],
        "player_name": ["Reference", "Broad match", "Modern match"] * 2,
        "season": ["2025-26"] * 6,
        "profile": ["Broad history"] * 3 + ["Modern detailed"] * 3,
        "position": ["G"] * 6,
        "dpm": [3.0] * 6,
        "o_dpm": [2.0] * 6,
        "d_dpm": [1.0] * 6,
        "eligible": [True] * 6,
        "lens": ["Offensive"] * 6,
        "method": ["Denoising autoencoder"] * 6,
        "embedding_00": [1.0, .99, 0.0, 1.0, 0.0, .99],
        "embedding_01": [0.0, .1, 1.0, 0.0, 1.0, .1],
    })
    broad = find_style_neighbors(
        frame, 1, lens="Offensive", method="Denoising autoencoder",
        season="2025-26", profile="Broad history", top_n=1,
    )
    modern = find_style_neighbors(
        frame, 1, lens="Offensive", method="Denoising autoencoder",
        season="2025-26", profile="Modern detailed", top_n=1,
    )
    assert broad.iloc[0]["player_name"] == "Broad match"
    assert modern.iloc[0]["player_name"] == "Modern match"
