"""Checks for model-derived Siamese pair explanations."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ball_ai.analytics.siamese_attribution import (
    exclusive_attribution_drivers,
    format_attribution_table,
    load_siamese_inference_artifacts,
    siamese_pair_attribution,
)
from ball_ai.config import settings


def test_numpy_attribution_reproduces_deployed_pair_score() -> None:
    features = pd.read_parquet(settings.historical_data_dir / "player_style_features.parquet")
    embeddings = pd.read_parquet(
        settings.historical_data_dir / "siamese_offensive_embeddings.parquet"
    ).set_index(["player_id", "season"])
    artifacts = load_siamese_inference_artifacts(
        Path("models/play_style/siamese_tabular_v5_inference.npz")
    )

    score, attribution = siamese_pair_attribution(
        features, artifacts, 201939, 1630538, "2024-25", "2025-26"
    )
    reference = embeddings.loc[(201939, "2024-25")]
    match = embeddings.loc[(1630538, "2025-26")]
    stable = [column for column in embeddings if column.startswith("siamese_embedding_")]
    detailed = [
        column for column in embeddings if column.startswith("detailed_siamese_embedding_")
    ]

    def angular(columns: list[str]) -> float:
        cosine = np.clip(
            reference[columns].to_numpy(float) @ match[columns].to_numpy(float), -1, 1
        )
        return float(1 - np.arccos(cosine) / np.pi)

    expected = 0.5 * angular(stable) + 0.5 * angular(detailed)
    assert score == pytest.approx(expected, abs=1e-6)
    assert attribution["Similarity contribution"].gt(0).any()
    assert attribution["Difference penalty"].gt(0).any()


def test_driver_lists_assign_overlaps_to_the_stronger_relative_effect() -> None:
    attribution = pd.DataFrame({
        "Feature": ["A", "B", "C", "D"],
        "Reference": [1.0, 2.0, 3.0, 4.0],
        "Match": [1.1, 2.2, 3.3, 4.4],
        "Similarity contribution": [0.6, 0.4, 0.2, 0.0],
        "Difference penalty": [0.1, 0.9, 0.0, 0.2],
    })

    similarity, difference = exclusive_attribution_drivers(attribution)

    assert similarity["Feature"].tolist() == ["A", "C"]
    assert difference["Feature"].tolist() == ["B", "D"]
    assert set(similarity["Feature"]).isdisjoint(difference["Feature"])
    assert set(format_attribution_table(
        attribution, "Similarity contribution", "One", "Two"
    )["Feature"]).isdisjoint(format_attribution_table(
        attribution, "Difference penalty", "One", "Two"
    )["Feature"])
