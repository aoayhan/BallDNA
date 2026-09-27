"""Reproduce the robustness checks used to promote the Siamese component."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ball_ai.analytics.play_style import (  # noqa: E402
    BROAD_OFFENSIVE_FEATURES,
    find_style_neighbors,
    fit_style_artifact,
    transform_style,
    transform_temporal_contrastive,
)
from ball_ai.config import settings  # noqa: E402
from scripts.evaluate_broad_v2_robustness import (  # noqa: E402
    BOOTSTRAP_SEASON,
    SPLIT_SEASONS,
    _eligible,
    _feature_rows,
    _cross_validated_knn,
    _load_games,
    _load_shots,
    _split_season,
    neighbor_overlap,
    retrieval_metrics,
)
from scripts.train_siamese_style_encoder import (  # noqa: E402
    fit_siamese_encoder,
    transform_siamese,
)


BOOTSTRAP_REPEATS = 10
REPRESENTATIVE_PLAYERS = {
    "Shai Gilgeous-Alexander": "2025-26",
    "Stephen Curry": "2025-26",
    "Nikola Jokic": "2025-26",
    "LeBron James": "2018-19",
    "Shaquille O'Neal": "2001-02",
}
HELD_OUT_TARGETS = {
    "rim_frequency": {
        "rim_frequency", "paint_frequency", "midrange_frequency",
        "three_point_frequency", "average_shot_distance",
    },
    "free_throw_attempt_rate": {"free_throw_attempt_rate"},
    "assists_per_36": {"assists_per_36", "assist_turnover_ratio"},
    "offensive_rebounds_per_36": {"offensive_rebounds_per_36"},
    "average_shot_distance": {
        "average_shot_distance", "rim_frequency", "paint_frequency",
        "midrange_frequency", "three_point_frequency",
    },
}


def _split_metrics(first: pd.DataFrame, second: pd.DataFrame, artifacts: dict) -> list[dict]:
    rows = []
    transforms = {
        "Denoising Player DNA": lambda frame: transform_style(
            artifacts["autoencoder"], frame, "Denoising autoencoder"
        ),
        "Temporal NCA": lambda frame: transform_temporal_contrastive(
            artifacts["temporal"], frame
        ),
        "Siamese encoder": lambda frame: transform_siamese(artifacts["siamese"], frame),
    }
    for model, transform in transforms.items():
        rows.append({
            "model": model,
            **retrieval_metrics(
                first["player_id"].to_numpy(),
                transform(first),
                second["player_id"].to_numpy(),
                transform(second),
            ),
        })
    return rows


def _bootstrap(root: Path, artifacts: dict) -> tuple[list[dict], pd.DataFrame]:
    games, shots = _load_games(root, BOOTSTRAP_SEASON), _load_shots(root, BOOTSTRAP_SEASON)
    full = _eligible(_feature_rows(games, shots, BOOTSTRAP_SEASON))
    attempts = full.set_index("player_id")["shot_attempts"]
    rows = []
    for repeat in range(BOOTSTRAP_REPEATS):
        sampled_games = (
            games.groupby("player_id", group_keys=False)
            .sample(frac=1, replace=True, random_state=42 + repeat)
        )
        sampled = _eligible(_feature_rows(sampled_games, shots, BOOTSTRAP_SEASON))
        common = np.intersect1d(full["player_id"], sampled["player_id"])
        base = full.set_index("player_id").loc[common].reset_index()
        boot = sampled.set_index("player_id").loc[common].reset_index()
        for model, transform in {
            "Denoising Player DNA": lambda frame: transform_style(
                artifacts["autoencoder"], frame, "Denoising autoencoder"
            ),
            "Temporal NCA": lambda frame: transform_temporal_contrastive(
                artifacts["temporal"], frame
            ),
            "Siamese encoder": lambda frame: transform_siamese(artifacts["siamese"], frame),
        }.items():
            left, right = transform(base), transform(boot)
            overlaps = neighbor_overlap(left, right)
            cosine = np.sum(left * right, axis=1)
            rows.extend({
                "repeat": repeat,
                "model": model,
                "player_id": int(player_id),
                "embedding_cosine": float(similarity),
                "top_10_neighbor_jaccard": float(jaccard),
                "shot_attempts": int(attempts.loc[player_id]),
            } for player_id, similarity, jaccard in zip(common, cosine, overlaps))
        print(f"Bootstrap repeat {repeat + 1}/{BOOTSTRAP_REPEATS}", flush=True)
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("model", as_index=False)
        .agg(
            mean_embedding_cosine=("embedding_cosine", "mean"),
            p10_embedding_cosine=("embedding_cosine", lambda values: values.quantile(.10)),
            mean_top_10_neighbor_jaccard=("top_10_neighbor_jaccard", "mean"),
            p10_top_10_neighbor_jaccard=(
                "top_10_neighbor_jaccard", lambda values: values.quantile(.10)
            ),
        )
    )
    return summary.to_dict(orient="records"), frame


def _player_checks(features: pd.DataFrame, embeddings: pd.DataFrame, artifacts: dict) -> dict:
    eligible = features["offensive_eligible"].fillna(False)
    cohort = features.loc[eligible].reset_index(drop=True)
    siamese_values = transform_siamese(artifacts["siamese"], cohort)
    key_to_index = {
        (int(row.player_id), row.season): index
        for index, row in enumerate(cohort.itertuples())
    }
    checks = {}
    for name, season in REPRESENTATIVE_PLAYERS.items():
        reference = cohort.loc[cohort["player_name"].eq(name) & cohort["season"].eq(season)]
        if reference.empty:
            checks[name] = {"season": season, "frozen_v1": [], "siamese_ensemble": []}
            continue
        player_id = int(reference.iloc[0]["player_id"])
        neighbors = find_style_neighbors(
            embeddings,
            player_id,
            lens="Offensive",
            method="Denoising autoencoder",
            profile="Broad history",
            season=season,
            top_n=20_000,
            features=features,
            artifact=artifacts["autoencoder"],
            presence_weight=.40,
            temporal_artifact=artifacts["temporal"],
            temporal_weight=.30,
        )
        reference_vector = siamese_values[key_to_index[(player_id, season)]]
        neighbors = neighbors.copy()
        neighbors["siamese_similarity"] = [
            1 - np.arccos(np.clip(
                reference_vector
                @ siamese_values[key_to_index[(int(row.player_id), row.season)]],
                -1,
                1,
            )) / np.pi
            for row in neighbors.itertuples()
        ]
        neighbors["siamese_ensemble_score"] = (
            .60 * neighbors["similarity_score"] + .40 * neighbors["siamese_similarity"]
        )
        columns = ["player_name", "season", "similarity_score", "siamese_ensemble_score"]
        frozen = neighbors.drop_duplicates("player_id").head(5)[columns]
        challenger = (
            neighbors.sort_values("siamese_ensemble_score", ascending=False)
            .drop_duplicates("player_id")
            .head(5)[columns]
        )
        checks[name] = {
            "season": season,
            "frozen_v1": frozen.to_dict(orient="records"),
            "siamese_ensemble": challenger.to_dict(orient="records"),
        }
    return checks


def _held_out_behavior(features: pd.DataFrame) -> list[dict]:
    """Test whether embeddings predict behavior deliberately removed before training."""

    rows = []
    cohort_mask = features["offensive_eligible"].fillna(False) & features["season"].eq(
        "2024-25"
    )
    for target, excluded in HELD_OUT_TARGETS.items():
        names = [name for name in BROAD_OFFENSIVE_FEATURES if name not in excluded]
        baseline = fit_style_artifact(
            features,
            "Offensive",
            profile="Broad history",
            training_end="2023-24",
            latent_dimensions=min(16, len(names)),
            feature_names=names,
        )
        siamese = fit_siamese_encoder(
            features,
            training_end="2023-24",
            embedding_dimensions=min(32, len(names)),
            temperature=.07,
            feature_names=names,
        )
        cohort = features.loc[cohort_mask & features[target].notna()].reset_index(drop=True)
        actual = cohort[target].to_numpy(dtype=float)
        for model, values in (
            ("Denoising Player DNA", transform_style(baseline, cohort, "Denoising autoencoder")),
            ("Siamese encoder", transform_siamese(siamese, cohort)),
        ):
            predicted, mean_predicted = _cross_validated_knn(values, actual)
            scale = float(np.std(actual)) or 1
            rows.append({
                "target": target,
                "model": model,
                "eligible_rows": len(cohort),
                "mae": float(mean_absolute_error(actual, predicted)),
                "normalized_mae": float(mean_absolute_error(actual, predicted) / scale),
                "r2": float(r2_score(actual, predicted)),
                "mean_baseline_mae": float(mean_absolute_error(actual, mean_predicted)),
            })
        print(f"Held-out behavior {target}", flush=True)
    return rows


def main() -> int:
    root = settings.historical_data_dir
    artifacts = {
        "autoencoder": joblib.load(root / "play_style_models.joblib")["Broad history"]["Offensive"],
        "temporal": joblib.load(
            root / "model_registry/play_style/candidates/temporal_contrastive_v1/model.joblib"
        ),
        "siamese": joblib.load(
            root / "model_registry/play_style/candidates/siamese_v1/model.joblib"
        ),
    }
    split_rows = []
    for season in SPLIT_SEASONS:
        print(f"Split season {season}", flush=True)
        first, second = _split_season(root, season)
        split_rows.extend({"season": season, **row} for row in _split_metrics(
            first, second, artifacts
        ))
    bootstrap, bootstrap_rows = _bootstrap(root, artifacts)
    features = pd.read_parquet(root / "player_style_features.parquet")
    embeddings = pd.read_parquet(root / "player_style_embeddings.parquet")
    players = _player_checks(features, embeddings, artifacts)
    held_out = _held_out_behavior(features)
    fold_report = json.loads(
        (ROOT / "models/play_style/siamese_tabular_v1_summary.json").read_text()
    )["holdout_evaluations"]
    fold_frame = pd.DataFrame(fold_report)
    pivot = fold_frame.pivot(index="fold", columns="model")
    fold_deltas = pd.DataFrame({
        metric: pivot[metric]["Siamese ensemble"] - pivot[metric]["Frozen v1"]
        for metric in ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
    }).reset_index()
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": "siamese-tabular-v1",
        "production_changed": True,
        "fold_deltas": fold_deltas.to_dict(orient="records"),
        "fold_wins": {
            metric: int((fold_deltas[metric] > 0).sum())
            for metric in ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
        },
        "split_season": split_rows,
        "bootstrap": bootstrap,
        "held_out_behavior": held_out,
        "representative_player_checks": players,
    }
    local = root / "model_registry/play_style/candidates/siamese_v1"
    (local / "robustness_evaluation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    bootstrap_rows.to_parquet(
        local / "bootstrap_player_stability.parquet", index=False, compression="snappy"
    )
    tracked = ROOT / "models/play_style/siamese_tabular_v1_robustness.json"
    tracked.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nFold wins\n", payload["fold_wins"])
    print("\nSplit-season means\n", pd.DataFrame(split_rows).groupby("model")[
        ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
    ].mean().to_string())
    print("\nBootstrap\n", pd.DataFrame(bootstrap).to_string(index=False))
    print("\nHeld-out behavior\n", pd.DataFrame(held_out).to_string(index=False))
    for name, result in players.items():
        print(f"\n{name} · {result['season']}")
        print("Siamese:", [row["player_name"] for row in result["siamese_ensemble"]])
    print(f"Saved {tracked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
