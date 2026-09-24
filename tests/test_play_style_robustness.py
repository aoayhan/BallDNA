"""Small checks for the Broad History robustness diagnostics."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_broad_v2_robustness import neighbor_overlap, retrieval_metrics


def test_retrieval_metrics_rank_the_same_player() -> None:
    result = retrieval_metrics(
        np.array([1, 2]), np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([2, 1]), np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
    assert result["recall_at_1"] == 1.0
    assert result["mean_self_margin"] == 1.0


def test_neighbor_overlap_is_one_for_identical_embeddings() -> None:
    values = np.eye(4)
    assert np.all(neighbor_overlap(values, values, top_n=2) == 1.0)
