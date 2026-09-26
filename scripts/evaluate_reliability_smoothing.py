"""Evaluate reliability-smoothed Broad History offense without changing production."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ball_ai.config import settings  # noqa: E402
from scripts.evaluate_play_style_rolling import ROLLING_FOLDS  # noqa: E402
from scripts.train_temporal_contrastive_encoder import (  # noqa: E402
    ADDITIONAL_HOLDOUT_FOLDS,
    HOLDOUT_FOLDS,
    _fold_scores,
    _metrics,
)


SHOT_FREQUENCIES = [
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
]
ATTEMPT_RATES = ["three_point_attempt_rate", "free_throw_attempt_rate"]
DEVELOPMENT_FOLDS = ROLLING_FOLDS[1:-1]
LATEST_TEST_FOLD = ROLLING_FOLDS[-1]
PRIOR_ATTEMPT_CANDIDATES = [0, 25, 50, 100, 200]
TEMPORAL_WEIGHT = 0.30


def reliability_smooth(features: pd.DataFrame, prior_attempts: float) -> pd.DataFrame:
    """Shrink noisy rate features toward attempt-weighted season means."""

    if prior_attempts < 0:
        raise ValueError("prior_attempts must be non-negative")
    output = features.copy()
    if prior_attempts == 0:
        return output
    for columns, denominator in (
        (SHOT_FREQUENCIES, "shot_attempts"),
        (ATTEMPT_RATES, "field_goal_attempts_total"),
    ):
        attempts = pd.to_numeric(output[denominator], errors="coerce")
        for column in columns:
            values = pd.to_numeric(output[column], errors="coerce")
            weighted = values * attempts
            valid_attempts = attempts.where(values.notna())
            league_rate = weighted.groupby(output["season"]).transform(
                "sum"
            ) / valid_attempts.groupby(output["season"]).transform("sum")
            smoothed = (weighted + prior_attempts * league_rate) / (attempts + prior_attempts)
            output[column] = smoothed.where(values.notna() & attempts.gt(0), values)
    return output


def _evaluate(
    features: pd.DataFrame,
    folds: list[tuple[str, str, str]],
    label: str,
) -> pd.DataFrame:
    rows = []
    for training_end, query_season, candidate_season in folds:
        print(f"{label}: {query_season} -> {candidate_season}", flush=True)
        query, candidates, hybrid, temporal = _fold_scores(
            features, training_end, query_season, candidate_season
        )
        rows.append({
            "set": label,
            "fold": f"{query_season} -> {candidate_season}",
            **_metrics(
                (1 - TEMPORAL_WEIGHT) * hybrid + TEMPORAL_WEIGHT * temporal,
                query,
                candidates,
            ),
        })
    return pd.DataFrame(rows)


def _summary(rows: pd.DataFrame) -> dict:
    return {
        metric: float(rows[metric].mean())
        for metric in ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
    }


def main() -> int:
    features = pd.read_parquet(
        settings.historical_data_dir
        / "model_registry/play_style/v1/player_style_features.parquet"
    )
    development_rows = []
    for prior_attempts in PRIOR_ATTEMPT_CANDIDATES:
        candidate = reliability_smooth(features, prior_attempts)
        evaluated = _evaluate(
            candidate, DEVELOPMENT_FOLDS, f"development alpha={prior_attempts}"
        )
        evaluated["prior_attempts"] = prior_attempts
        development_rows.append(evaluated)
    development = pd.concat(development_rows, ignore_index=True)
    development_summary = (
        development.groupby("prior_attempts", as_index=False)[
            ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
        ]
        .mean()
        .sort_values("mean_reciprocal_rank", ascending=False)
    )
    selected_alpha = float(development_summary.iloc[0]["prior_attempts"])
    selected_features = reliability_smooth(features, selected_alpha)
    holdouts = _evaluate(
        selected_features,
        [*HOLDOUT_FOLDS, *ADDITIONAL_HOLDOUT_FOLDS],
        "all_19_holdouts",
    )
    latest = _evaluate(selected_features, [LATEST_TEST_FOLD], "latest_test")
    baseline = json.loads(
        (ROOT / "models/play_style/temporal_contrastive_v1_summary.json").read_text()
    )["combined_holdout_summary"]
    baseline = next(row for row in baseline if row["model"] == "Temporal ensemble")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": "reliability-smoothing-v1",
        "status": "offline_challenger",
        "production_changed": False,
        "selected_prior_attempts": selected_alpha,
        "smoothed_features": [*SHOT_FREQUENCIES, *ATTEMPT_RATES],
        "development_summary": development_summary.to_dict(orient="records"),
        "all_19_holdouts": _summary(holdouts),
        "frozen_v1_all_19_holdouts": {
            key: baseline[key]
            for key in ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
        },
        "latest_test": _summary(latest),
        "holdout_evaluations": holdouts.to_dict(orient="records"),
    }
    output = ROOT / "models/play_style/reliability_smoothing_v1_summary.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nDevelopment\n", development_summary.to_string(index=False))
    print("\nAll 19 holdouts\n", pd.DataFrame([payload["all_19_holdouts"]]).to_string(index=False))
    print("\nFrozen v1\n", pd.DataFrame([payload["frozen_v1_all_19_holdouts"]]).to_string(index=False))
    print("\nLatest test\n", pd.DataFrame([payload["latest_test"]]).to_string(index=False))
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
