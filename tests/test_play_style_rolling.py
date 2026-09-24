"""Checks that Player DNA rolling fits stop before both retrieval seasons."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_play_style_rolling import HISTORICAL_OFFENSIVE_FOLDS, RECENT_FOLDS


def test_rolling_folds_train_before_both_retrieval_seasons() -> None:
    for training_end, query_season, candidate_season in [
        *HISTORICAL_OFFENSIVE_FOLDS,
        *RECENT_FOLDS,
    ]:
        train_year = int(training_end[:4])
        query_year = int(query_season[:4])
        candidate_year = int(candidate_season[:4])

        assert train_year + 1 == candidate_year
        assert candidate_year + 1 == query_year


def test_broad_offensive_backtest_reaches_the_1990s() -> None:
    assert HISTORICAL_OFFENSIVE_FOLDS[0] == ("1996-97", "1998-99", "1997-98")
