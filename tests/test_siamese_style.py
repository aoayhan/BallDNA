"""Minimal correctness checks for the Siamese Player DNA challenger."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from scripts.train_siamese_style_encoder import (
    adjacent_pair_indices,
    supervised_contrastive_loss,
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
