"""Promote the validated wider Siamese encoder without overwriting v4 rollback files."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ball_ai.analytics.play_style import top_siamese_style_pairs  # noqa: E402
from ball_ai.config import settings  # noqa: E402
from scripts.train_siamese_multiview_challenger import (  # noqa: E402
    DETAIL_WEIGHT,
    DETAILED_FEATURES,
    STABLE_FEATURES,
    transform,
)
from scripts.train_siamese_style_encoder import (  # noqa: E402
    export_numpy_inference_artifacts,
)


VERSION = "v5"
CANDIDATE = (
    settings.historical_data_dir
    / "model_registry/play_style/candidates/siamese_wider_v1"
)


def _embedding_frame(stable: dict, detailed: dict, features: pd.DataFrame) -> pd.DataFrame:
    eligible = features.loc[features["offensive_eligible"].fillna(False)].copy()
    stable_vectors = transform(stable, eligible)
    detailed_vectors = transform(detailed, eligible)
    output = eligible[["player_id", "season"]].reset_index(drop=True)
    for index in range(stable_vectors.shape[1]):
        output[f"siamese_embedding_{index:02d}"] = stable_vectors[:, index]
        output[f"detailed_siamese_embedding_{index:02d}"] = detailed_vectors[:, index]
    return output


def main() -> int:
    stable = joblib.load(CANDIDATE / "siamese_model.joblib")
    detailed = joblib.load(CANDIDATE / "detailed_siamese_model.joblib")
    if tuple(stable["hidden"]) != (128, 64) or tuple(detailed["hidden"]) != (128, 64):
        raise ValueError("Promotion requires the validated 128 -> 64 wider artifacts.")

    features = pd.read_parquet(
        settings.historical_data_dir
        / "model_registry/play_style/v1/player_style_features.parquet"
    )
    embeddings = _embedding_frame(stable, detailed, features)
    embeddings_path = settings.historical_data_dir / "siamese_offensive_embeddings.parquet"
    embeddings.to_parquet(embeddings_path, index=False)

    export_numpy_inference_artifacts(
        stable,
        detailed,
        ROOT / "models/play_style/siamese_tabular_v5_inference.npz",
        DETAIL_WEIGHT,
    )

    metadata = features[[
        "player_id", "season", "player_name", "position", "shot_attempts",
    ]]
    impact_path = settings.historical_data_dir / "darko_dpm.parquet"
    impact_columns = ["player_id", "season", "dpm", "o_dpm", "d_dpm"]
    impact = (
        pd.read_parquet(impact_path, columns=impact_columns)
        if impact_path.exists()
        else features[impact_columns]
    ).drop_duplicates(["player_id", "season"], keep="last")
    metadata = metadata.merge(
        impact,
        on=["player_id", "season"],
        how="left",
        validate="one_to_one",
    )
    leaderboard = top_siamese_style_pairs(
        embeddings, metadata, top_n=100, detail_weight=DETAIL_WEIGHT
    )
    leaderboard.to_parquet(
        settings.historical_data_dir / "style_similarity_leaderboard.parquet",
        index=False,
    )

    evaluation = json.loads(
        (ROOT / "models/play_style/siamese_wider_v1_multiseed_folds.json").read_text()
    )
    robustness = json.loads(
        (ROOT / "models/play_style/siamese_wider_v1_robustness.json").read_text()
    )
    wider = evaluation["summary"]["model_means"]["Wider"]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": "siamese-tabular-v5-wider",
        "status": "promoted_broad_offense_component",
        "production_changed": True,
        "architecture": "shared 128 -> 64 MLPs with 32-dimensional L2-normalized embeddings",
        "objective": "Multi-positive supervised contrastive InfoNCE",
        "positive_pairs": "Adjacent eligible seasons and stochastic feature views",
        "stable_feature_count": len(STABLE_FEATURES),
        "detailed_feature_count": len(DETAILED_FEATURES),
        "detail_coverage_start": "2007-08",
        "input_exclusions": [
            "player identity", "player name", "team", "position", "height", "weight",
            "shooting efficiency", "DARKO impact",
        ],
        "selected_embedding_dimensions": 32,
        "selected_temperature": .07,
        "selected_detail_weight": DETAIL_WEIGHT,
        "evaluation_protocol": "five seeds across 19 chronological holdout folds",
        "all_19_holdouts": {"coverage_aware_ensemble": wider},
        "paired_fold_evidence": evaluation["summary"]["paired_fold_evidence"],
        "latest_fold_seed_summary": robustness["seed_summary"],
        "promotion_gate_pass": evaluation["promotion_gate_pass"],
    }
    (ROOT / "models/play_style/siamese_tabular_v5_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Promoted {len(embeddings):,} embeddings and {len(leaderboard):,} pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
