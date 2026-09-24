"""Tests for the structured player similarity baseline."""

from __future__ import annotations

import pandas as pd
import pytest

from ball_ai.analytics.similarity import (
    SIMILARITY_FEATURES,
    contribution_table,
    find_similar_players,
    percentile_embeddings,
    similarity_matrix,
)
from ball_ai.analytics.shot_profile import SHOT_PROFILE_FEATURES


def _profiles() -> pd.DataFrame:
    rows = []
    for player_id, offset in [(1, 0.0), (2, 0.1), (3, 5.0)]:
        row = {"player_id": player_id, "player_name": f"P{player_id}"}
        row.update({feature: index + offset for index, feature in enumerate(SIMILARITY_FEATURES)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_similarity_matrix_is_symmetric_with_unit_diagonal() -> None:
    matrix = similarity_matrix(_profiles())
    assert matrix.loc[1, 2] == pytest.approx(matrix.loc[2, 1])
    assert matrix.loc[1, 1] == pytest.approx(1.0)


def test_similar_players_excludes_selected_player() -> None:
    result = find_similar_players(_profiles(), player_id=1, top_n=2)
    assert 1 not in result["player_id"].tolist()
    assert len(result) == 2


def test_unknown_player_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Unknown player_id"):
        find_similar_players(_profiles(), player_id=99)


def test_percentile_embedding_and_similarity_are_bounded() -> None:
    embeddings = percentile_embeddings(_profiles())
    result = find_similar_players(
        _profiles(), player_id=1, preset="Scoring engine", top_n=2
    )

    assert embeddings.min().min() >= 0
    assert embeddings.max().max() <= 1
    assert result["similarity_score"].between(0, 1).all()


def test_explanation_contributions_sum_to_one() -> None:
    neighbor = find_similar_players(_profiles(), player_id=1, top_n=1).iloc[0]
    explanation = contribution_table(neighbor)

    assert explanation["Distance contribution"].sum() == pytest.approx(1.0)


def test_available_shot_features_extend_the_embedding() -> None:
    profiles = _profiles()
    for index, feature in enumerate(SHOT_PROFILE_FEATURES):
        profiles[feature] = [index + .1, index + .2, index + .9]
    profiles["shot_attempts"] = [200, 180, 10]
    result = find_similar_players(
        profiles, player_id=1, top_n=2, minimum_shots=100
    )

    assert result["player_id"].tolist() == [2]
    assert "contribution__floater_frequency" in result
