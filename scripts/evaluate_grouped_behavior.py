"""Evaluate behavioral group weights without changing the deployed model."""

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
for path in (ROOT, SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ball_ai.analytics.play_style import (  # noqa: E402
    build_embedding_frame,
    fit_style_artifact,
    fit_temporal_contrastive_artifact,
    transform_temporal_contrastive,
)
from ball_ai.config import settings  # noqa: E402
from scripts.evaluate_play_style_rolling import ROLLING_FOLDS  # noqa: E402
from scripts.train_temporal_contrastive_encoder import (  # noqa: E402
    ADDITIONAL_HOLDOUT_FOLDS,
    HOLDOUT_FOLDS,
    _cohorts,
    _hybrid_scores,
    _metrics,
)


FEATURE_GROUPS = {
    "shot_location": [
        "rim_frequency", "paint_frequency", "midrange_frequency",
        "three_point_frequency", "average_shot_distance",
    ],
    "shot_action": [
        "dunk_frequency", "layup_frequency", "floater_frequency",
        "hook_frequency", "pull_up_frequency", "step_back_frequency",
    ],
    "creation": [
        "field_goal_attempts_per_36", "three_point_attempt_rate",
        "free_throw_attempt_rate", "estimated_used_possessions_per_36",
    ],
    "playmaking": [
        "assists_per_36", "turnovers_per_36", "assist_turnover_ratio",
    ],
    "offensive_rebounding": ["offensive_rebounds_per_36"],
}
GROUP_BUDGETS = {
    "deployed_uniform": None,
    "role_balanced": {
        "shot_location": .25, "shot_action": .25, "creation": .20,
        "playmaking": .25, "offensive_rebounding": .05,
    },
    "creator_priority": {
        "shot_location": .20, "shot_action": .20, "creation": .25,
        "playmaking": .30, "offensive_rebounding": .05,
    },
    "shot_priority": {
        "shot_location": .30, "shot_action": .30, "creation": .15,
        "playmaking": .20, "offensive_rebounding": .05,
    },
}
DEVELOPMENT_FOLDS = ROLLING_FOLDS[1:-1]
LATEST_TEST_FOLD = ROLLING_FOLDS[-1]
TEMPORAL_WEIGHT = .30


def group_feature_weights(budgets: dict[str, float]) -> dict[str, float]:
    """Give each feature group its requested squared-distance budget."""

    if set(budgets) != set(FEATURE_GROUPS) or not np.isclose(sum(budgets.values()), 1):
        raise ValueError("Group budgets must cover every group and sum to one.")
    weights = {
        feature: np.sqrt(budgets[group] / len(features))
        for group, features in FEATURE_GROUPS.items()
        for feature in features
    }
    mean = float(np.mean(list(weights.values())))
    return {feature: float(weight / mean) for feature, weight in weights.items()}


def _fold_scores(
    features: pd.DataFrame,
    training_end: str,
    query_season: str,
    candidate_season: str,
    feature_weights: dict[str, float] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    query, candidates = _cohorts(features, query_season, candidate_season)
    autoencoder = fit_style_artifact(
        features,
        "Offensive",
        profile="Broad history",
        training_end=training_end,
        latent_dimensions=16,
        feature_weights=feature_weights,
    )
    embeddings = build_embedding_frame(features, autoencoder, "Denoising autoencoder")
    hybrid = _hybrid_scores(features, autoencoder, embeddings, query, candidates)
    temporal = fit_temporal_contrastive_artifact(
        features, training_end=training_end, latent_dimensions=16
    )
    temporal_query = transform_temporal_contrastive(temporal, query)
    temporal_candidates = transform_temporal_contrastive(temporal, candidates)
    temporal_cosine = np.clip(temporal_query @ temporal_candidates.T, -1, 1)
    temporal_score = 1 - np.arccos(temporal_cosine) / np.pi
    ensemble = (1 - TEMPORAL_WEIGHT) * hybrid + TEMPORAL_WEIGHT * temporal_score
    return query, candidates, ensemble


def _evaluate(
    features: pd.DataFrame,
    folds: list[tuple[str, str, str]],
    label: str,
    feature_weights: dict[str, float] | None,
) -> pd.DataFrame:
    rows = []
    for training_end, query_season, candidate_season in folds:
        print(f"{label}: {query_season} -> {candidate_season}", flush=True)
        query, candidates, scores = _fold_scores(
            features, training_end, query_season, candidate_season, feature_weights
        )
        rows.append({
            "set": label,
            "fold": f"{query_season} -> {candidate_season}",
            **_metrics(scores, query, candidates),
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
    resolved_weights = {}
    for variant, budgets in GROUP_BUDGETS.items():
        weights = group_feature_weights(budgets) if budgets else None
        resolved_weights[variant] = weights
        evaluated = _evaluate(
            features, DEVELOPMENT_FOLDS, f"development {variant}", weights
        )
        evaluated["variant"] = variant
        development_rows.append(evaluated)
    development = pd.concat(development_rows, ignore_index=True)
    development_summary = (
        development.groupby("variant", as_index=False)[
            ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
        ]
        .mean()
        .sort_values("mean_reciprocal_rank", ascending=False)
    )
    selected = str(development_summary.iloc[0]["variant"])
    holdouts = _evaluate(
        features,
        [*HOLDOUT_FOLDS, *ADDITIONAL_HOLDOUT_FOLDS],
        "all_19_holdouts",
        resolved_weights[selected],
    )
    latest = _evaluate(
        features, [LATEST_TEST_FOLD], "latest_test", resolved_weights[selected]
    )
    baseline = json.loads(
        (ROOT / "models/play_style/temporal_contrastive_v1_summary.json").read_text()
    )["combined_holdout_summary"]
    baseline = next(row for row in baseline if row["model"] == "Temporal ensemble")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": "grouped-behavior-v1",
        "status": "offline_challenger",
        "production_changed": False,
        "selected_variant": selected,
        "group_budgets": GROUP_BUDGETS,
        "selected_feature_weights": resolved_weights[selected],
        "development_summary": development_summary.to_dict(orient="records"),
        "all_19_holdouts": _summary(holdouts),
        "frozen_v1_all_19_holdouts": {
            key: baseline[key]
            for key in ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
        },
        "latest_test": _summary(latest),
        "holdout_evaluations": holdouts.to_dict(orient="records"),
    }
    output = ROOT / "models/play_style/grouped_behavior_v1_summary.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nDevelopment\n", development_summary.to_string(index=False))
    print("\nAll 19 holdouts\n", pd.DataFrame([payload["all_19_holdouts"]]).to_string(index=False))
    print("\nFrozen v1\n", pd.DataFrame([payload["frozen_v1_all_19_holdouts"]]).to_string(index=False))
    print("\nLatest test\n", pd.DataFrame([payload["latest_test"]]).to_string(index=False))
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
