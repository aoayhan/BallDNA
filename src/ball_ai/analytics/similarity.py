"""Explainable, role-aware player retrieval from structured season features.

The feature vector is intentionally deterministic: pandas creates percentile
embeddings, a role preset supplies feature weights, and weighted Euclidean
distance retrieves the nearest profiles. A deterministic summary may explain the result,
but it never chooses or changes the neighbors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ball_ai.analytics.shot_profile import SHOT_PROFILE_FEATURES


SIMILARITY_FEATURES = [
    "field_goal_percentage",
    "three_point_percentage",
    "free_throw_percentage",
    "rebounds_per_game",
    "assists_per_game",
    "steals_per_game",
    "blocks_per_game",
    "turnovers_per_game",
    "points_per_game",
]

FEATURE_LABELS = {
    "field_goal_percentage": "FG%",
    "three_point_percentage": "3P%",
    "free_throw_percentage": "FT%",
    "rebounds_per_game": "TRB",
    "assists_per_game": "AST",
    "steals_per_game": "STL",
    "blocks_per_game": "BLK",
    "turnovers_per_game": "TOV",
    "points_per_game": "PTS",
    "rim_frequency": "Rim frequency",
    "paint_frequency": "Paint frequency",
    "midrange_frequency": "Midrange frequency",
    "three_point_frequency": "3-point frequency",
    "dunk_frequency": "Dunk frequency",
    "layup_frequency": "Layup frequency",
    "floater_frequency": "Floater frequency",
    "hook_frequency": "Hook frequency",
    "pull_up_frequency": "Pull-up frequency",
    "step_back_frequency": "Step-back frequency",
    "average_shot_distance": "Average shot distance",
    "shot_difficulty_index": "Shot difficulty index",
    "shot_making_above_expected": "Shot-making above expected",
}

FEATURE_FAMILIES = {
    "Efficiency": [
        "field_goal_percentage",
        "three_point_percentage",
        "free_throw_percentage",
    ],
    "Production": ["points_per_game", "rebounds_per_game"],
    "Creation": ["assists_per_game", "turnovers_per_game"],
    "Box-score defense": ["steals_per_game", "blocks_per_game"],
    "Shot selection": [
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
    ],
    "Shot context and results": [
        "average_shot_distance",
        "shot_difficulty_index",
        "shot_making_above_expected",
    ],
}

SHOT_ROLE_WEIGHTS = {
    "Balanced profile": {feature: 1.0 for feature in SHOT_PROFILE_FEATURES},
    "Scoring engine": {
        "pull_up_frequency": 1.5,
        "step_back_frequency": 1.5,
        "shot_difficulty_index": 1.5,
        "shot_making_above_expected": 2.0,
    },
    "Spacing wing": {
        "three_point_frequency": 4.0,
        "midrange_frequency": 1.5,
        "pull_up_frequency": 1.25,
        "step_back_frequency": 1.25,
        "average_shot_distance": 2.0,
        "shot_making_above_expected": 1.5,
    },
    "Primary creator": {
        "floater_frequency": 1.5,
        "pull_up_frequency": 3.0,
        "step_back_frequency": 2.5,
        "shot_difficulty_index": 2.0,
        "shot_making_above_expected": 1.5,
    },
    "Interior impact": {
        "rim_frequency": 4.0,
        "paint_frequency": 2.5,
        "dunk_frequency": 3.0,
        "layup_frequency": 2.0,
        "hook_frequency": 2.0,
        "average_shot_distance": 2.0,
        "shot_difficulty_index": 1.5,
    },
    "Two-way impact": {
        "rim_frequency": 1.5,
        "three_point_frequency": 1.5,
        "shot_making_above_expected": 1.5,
    },
}

# Presets alter distance, not player quality. A high-turnover player can still be
# similar to another high-turnover player because similarity is non-directional.
ROLE_PRESETS = {
    "Balanced profile": {
        "description": "Equal emphasis across box-score production, efficiency, creation, defense, shot selection, action type, and modeled shot context.",
        "weights": {feature: 1.0 for feature in SIMILARITY_FEATURES},
    },
    "Scoring engine": {
        "description": "Prioritizes scoring volume, shooting efficiency, self-created shot mix, difficulty, and shot making above expectation.",
        "weights": {
            "field_goal_percentage": 2.0,
            "three_point_percentage": 1.5,
            "free_throw_percentage": 1.0,
            "rebounds_per_game": 0.5,
            "assists_per_game": 1.0,
            "steals_per_game": 0.5,
            "blocks_per_game": 0.5,
            "turnovers_per_game": 1.0,
            "points_per_game": 3.0,
        },
    },
    "Spacing wing": {
        "description": "Prioritizes perimeter efficiency, three-point frequency, average distance, and supporting two-way indicators.",
        "weights": {
            "field_goal_percentage": 1.0,
            "three_point_percentage": 4.0,
            "free_throw_percentage": 1.5,
            "rebounds_per_game": 1.0,
            "assists_per_game": 0.75,
            "steals_per_game": 1.25,
            "blocks_per_game": 0.5,
            "turnovers_per_game": 0.75,
            "points_per_game": 1.5,
        },
    },
    "Primary creator": {
        "description": "Prioritizes assists, turnover profile, scoring load, pull-ups, step-backs, floaters, and modeled difficulty.",
        "weights": {
            "field_goal_percentage": 1.0,
            "three_point_percentage": 1.0,
            "free_throw_percentage": 1.0,
            "rebounds_per_game": 0.5,
            "assists_per_game": 4.0,
            "steals_per_game": 0.75,
            "blocks_per_game": 0.25,
            "turnovers_per_game": 3.0,
            "points_per_game": 2.0,
        },
    },
    "Interior impact": {
        "description": "Prioritizes rebounding, blocks, rim/paint frequency, dunks, layups, hooks, and interior efficiency.",
        "weights": {
            "field_goal_percentage": 2.0,
            "three_point_percentage": 0.25,
            "free_throw_percentage": 0.75,
            "rebounds_per_game": 3.0,
            "assists_per_game": 0.5,
            "steals_per_game": 0.75,
            "blocks_per_game": 3.0,
            "turnovers_per_game": 0.75,
            "points_per_game": 1.5,
        },
    },
    "Two-way impact": {
        "description": "Balances scoring and shot mix with the available steals, blocks, rebounding, and efficiency proxies.",
        "weights": {
            "field_goal_percentage": 1.5,
            "three_point_percentage": 1.25,
            "free_throw_percentage": 0.75,
            "rebounds_per_game": 1.5,
            "assists_per_game": 0.75,
            "steals_per_game": 3.0,
            "blocks_per_game": 2.5,
            "turnovers_per_game": 0.75,
            "points_per_game": 2.0,
        },
    },
}


def _validate_input(season_stats: pd.DataFrame) -> None:
    if list(season_stats.columns).count("player_id") != 1:
        raise ValueError("Similarity input must contain exactly one player_id column.")
    missing = set(SIMILARITY_FEATURES) - set(season_stats.columns)
    if missing:
        raise ValueError(f"Missing similarity features: {sorted(missing)}")
    if len(season_stats) < 2:
        raise ValueError("At least two players are required for similarity.")


def active_similarity_features(season_stats: pd.DataFrame) -> list[str]:
    """Return baseline plus sufficiently populated optional shot features."""

    optional = [
        feature
        for feature in SHOT_PROFILE_FEATURES
        if feature in season_stats
        and pd.to_numeric(season_stats[feature], errors="coerce").notna().sum() >= 2
    ]
    return [*SIMILARITY_FEATURES, *optional]


def percentile_embeddings(season_stats: pd.DataFrame) -> pd.DataFrame:
    """Return 0–1 league-percentile embeddings for the supported features."""

    _validate_input(season_stats)
    features = active_similarity_features(season_stats)
    values = season_stats[features].astype(float)
    values = values.fillna(values.median()).fillna(0.0)
    embeddings = values.rank(method="average", pct=True)
    embeddings.index = season_stats["player_id"].astype(int)
    return embeddings


def _weights(preset: str, features: list[str]) -> pd.Series:
    if preset not in ROLE_PRESETS:
        raise ValueError(f"Unknown role preset: {preset}")
    configured = {
        **ROLE_PRESETS[preset]["weights"],
        **SHOT_ROLE_WEIGHTS.get(preset, {}),
    }
    return pd.Series(configured, dtype=float).reindex(features).fillna(1.0)


def similarity_matrix(
    season_stats: pd.DataFrame,
    preset: str = "Balanced profile",
) -> pd.DataFrame:
    """Return pairwise 0–1 similarities from weighted percentile distance."""

    embeddings = percentile_embeddings(season_stats)
    weights = _weights(preset, embeddings.columns.tolist()).to_numpy()
    values = embeddings.to_numpy()
    differences = values[:, np.newaxis, :] - values[np.newaxis, :, :]
    distances = np.sqrt(
        np.sum(weights * differences**2, axis=2) / float(weights.sum())
    )
    scores = np.clip(1.0 - distances, 0.0, 1.0)
    return pd.DataFrame(scores, index=embeddings.index, columns=embeddings.index)


def positions_are_compatible(reference: str, candidate: str) -> bool:
    """Return whether broad position labels share at least one role token."""

    reference_tokens = {token for token in str(reference).split("-") if token != "N/A"}
    candidate_tokens = {token for token in str(candidate).split("-") if token != "N/A"}
    return bool(reference_tokens & candidate_tokens) or not reference_tokens


def find_similar_players(
    season_stats: pd.DataFrame,
    player_id: int,
    top_n: int = 5,
    preset: str = "Balanced profile",
    position_aware: bool = False,
    minimum_games: int = 0,
    minimum_shots: int = 0,
) -> pd.DataFrame:
    """Retrieve neighbors and expose every feature's distance contribution."""

    _validate_input(season_stats)
    all_player_ids = set(season_stats["player_id"].astype(int))
    if int(player_id) not in all_player_ids:
        raise ValueError(f"Unknown player_id: {player_id}")

    reference_row = season_stats.loc[
        season_stats["player_id"].astype(int).eq(int(player_id))
    ].iloc[0]
    cohort = season_stats.copy()
    if "games_played" in cohort and minimum_games:
        qualified = cohort["games_played"].astype(float).ge(float(minimum_games))
        cohort = cohort.loc[qualified | cohort["player_id"].astype(int).eq(int(player_id))]
    if "shot_attempts" in cohort and minimum_shots:
        qualified = cohort["shot_attempts"].fillna(0).astype(float).ge(float(minimum_shots))
        cohort = cohort.loc[qualified | cohort["player_id"].astype(int).eq(int(player_id))]
    if position_aware and "position" in cohort:
        compatible = cohort["position"].map(
            lambda value: positions_are_compatible(reference_row.get("position", "N/A"), value)
        )
        cohort = cohort.loc[compatible | cohort["player_id"].astype(int).eq(int(player_id))]
    if len(cohort) < 2:
        raise ValueError("No comparison candidates remain after applying the filters.")

    embeddings = percentile_embeddings(cohort)
    weights = _weights(preset, embeddings.columns.tolist())
    reference = embeddings.loc[int(player_id)]
    squared = embeddings.sub(reference, axis="columns").pow(2).mul(weights, axis="columns")
    weighted_distance = np.sqrt(squared.sum(axis=1) / float(weights.sum()))
    scores = (1.0 - weighted_distance).clip(0.0, 1.0)

    ranked_ids = (
        scores.drop(index=int(player_id))
        .sort_values(ascending=False)
        .head(int(top_n))
        .index
    )
    details = cohort.copy()
    details["player_id"] = details["player_id"].astype(int)
    details = details.set_index("player_id").loc[ranked_ids].copy()
    details["similarity_score"] = scores.loc[ranked_ids]
    details["distance"] = weighted_distance.loc[ranked_ids]

    for feature in embeddings.columns:
        details[f"reference_pct__{feature}"] = float(reference[feature])
        details[f"candidate_pct__{feature}"] = embeddings.loc[ranked_ids, feature]
        details[f"difference__{feature}"] = embeddings.loc[ranked_ids, feature].sub(
            float(reference[feature])
        ).abs()
        denominator = squared.loc[ranked_ids].sum(axis=1).replace(0, np.nan)
        details[f"contribution__{feature}"] = (
            squared.loc[ranked_ids, feature].div(denominator).fillna(0.0)
        )

    return details.reset_index()


def contribution_table(neighbor: pd.Series | dict) -> pd.DataFrame:
    """Return a human-readable explanation of one retrieved neighbor."""

    row = dict(neighbor)
    records = []
    features = [
        feature
        for feature in FEATURE_LABELS
        if f"contribution__{feature}" in row
    ]
    for feature in features:
        records.append(
            {
                "Feature": FEATURE_LABELS[feature],
                "Reference percentile": float(row[f"reference_pct__{feature}"]),
                "Match percentile": float(row[f"candidate_pct__{feature}"]),
                "Percentile gap": float(row[f"difference__{feature}"]),
                "Distance contribution": float(row[f"contribution__{feature}"]),
            }
        )
    return pd.DataFrame(records).sort_values("Distance contribution", ascending=False)
