"""Minimal correctness checks for the Siamese Player DNA challenger."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from scripts.train_siamese_style_encoder import (
    adjacent_pair_indices,
    supervised_contrastive_loss,
)
from scripts.train_siamese_multiview_challenger import (
    alternating_game_views,
    weighted_supervised_contrastive_loss,
)


def test_siamese_pairs_and_loss() -> None:
    frame = pd.DataFrame({
        "player_id": [1, 1, 1, 2, 2],
        "season": ["2020-21", "2021-22", "2023-24", "2020-21", "2022-23"],
    })
    assert adjacent_pair_indices(frame).tolist() == [[0, 1]]

    aligned = torch.tensor([[1.0, 0], [0, 1.0], [1.0, 0], [0, 1.0]])
    labels = torch.tensor([1, 2, 1, 2])
    shuffled = aligned[[0, 1, 1, 0]]
    assert supervised_contrastive_loss(aligned, labels, .1) < supervised_contrastive_loss(
        shuffled, labels, .1
    )


def test_multiview_order_and_weighted_loss() -> None:
    games = pd.DataFrame({
        "player_id": [1, 2, 1, 1, 2],
        "game_date": pd.to_datetime([
            "2024-01-03", "2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01"
        ]),
        "game_id": [3, 2, 1, 2, 1],
    })
    ordered = alternating_game_views(games)
    assert ordered.loc[ordered["player_id"].eq(1), "view"].tolist() == ["odd", "even", "odd"]
    assert ordered.loc[ordered["player_id"].eq(2), "view"].tolist() == ["odd", "even"]

    aligned = torch.tensor([[1.0, 0], [0, 1.0], [1.0, 0], [0, 1.0]])
    labels = torch.tensor([1, 2, 1, 2])
    weights = torch.ones(4)
    assert torch.isclose(
        weighted_supervised_contrastive_loss(aligned, labels, .1, weights),
        weighted_supervised_contrastive_loss(aligned, labels, .1),
    )
