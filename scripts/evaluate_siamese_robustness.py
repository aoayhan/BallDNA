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
    MODERN_MOVE_FEATURES,
    SIAMESE_DETAIL_WEIGHT,
    SIAMESE_ENSEMBLE_WEIGHT,
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
    _coverage_aware_scores,
    fit_siamese_encoder,
    transform_siamese,
)
from scripts.train_temporal_contrastive_encoder import _cohorts  # noqa: E402


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


def _score_matrix_metrics(
    query_ids: np.ndarray, scores: np.ndarray, candidate_ids: np.ndarray
) -> dict:
    shared = np.intersect1d(query_ids, candidate_ids)
    query_mask = np.isin(query_ids, shared)
    candidate_mask = np.isin(candidate_ids, shared)
    query_ids = query_ids[query_mask]
    candidate_ids = candidate_ids[candidate_mask]
    scores = scores[np.ix_(query_mask, candidate_mask)]
    ranks = np.asarray([
        np.flatnonzero(candidate_ids[np.argsort(-scores[index])] == int(player_id))[0] + 1
        for index, player_id in enumerate(query_ids)
    ])
    return {
        "eligible_queries": len(ranks),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_3": float(np.mean(ranks <= 3)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mean_reciprocal_rank": float(np.mean(1 / ranks)),
        "median_rank": float(np.median(ranks)),
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
    _, score = _coverage_aware_scores(
        artifacts["siamese"], artifacts["detailed_siamese"],
        first, second, SIAMESE_DETAIL_WEIGHT,
    )
    rows.append({
        "model": "Coverage-aware Siamese",
        **_score_matrix_metrics(
            first["player_id"].to_numpy(), score,
            second["player_id"].to_numpy(),
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
            "Coverage-aware Siamese": lambda frame: np.concatenate([
                np.sqrt(1 - SIAMESE_DETAIL_WEIGHT)
                * transform_siamese(artifacts["siamese"], frame),
                np.sqrt(SIAMESE_DETAIL_WEIGHT)
                * transform_siamese(artifacts["detailed_siamese"], frame),
            ], axis=1),
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


def _player_checks(
    features: pd.DataFrame,
    embeddings: pd.DataFrame,
    siamese_embeddings: pd.DataFrame,
) -> dict:
    eligible = features["offensive_eligible"].fillna(False)
    cohort = features.loc[eligible].reset_index(drop=True)
    checks = {}
    for name, season in REPRESENTATIVE_PLAYERS.items():
        reference = cohort.loc[cohort["player_name"].eq(name) & cohort["season"].eq(season)]
        if reference.empty:
            checks[name] = {"season": season, "stable_siamese": [], "coverage_aware": []}
            continue
        player_id = int(reference.iloc[0]["player_id"])
        kwargs = dict(
            lens="Offensive", method="Denoising autoencoder", profile="Broad history",
            season=season, top_n=20_000, siamese_embeddings=siamese_embeddings,
            siamese_weight=1.0, unique_players=True,
        )
        stable = find_style_neighbors(
            embeddings, player_id, siamese_detail_weight=0.0, **kwargs
        ).head(5)
        challenger = find_style_neighbors(embeddings, player_id, **kwargs).head(5)
        columns = ["player_name", "season", "similarity_score"]
        checks[name] = {
            "season": season,
            "stable_siamese": stable[columns].to_dict(orient="records"),
            "coverage_aware": challenger[columns].to_dict(orient="records"),
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
        detailed_names = [
            name for name in [*BROAD_OFFENSIVE_FEATURES, *MODERN_MOVE_FEATURES]
            if name not in excluded
        ]
        detailed = fit_siamese_encoder(
            features,
            training_start="2007-08",
            training_end="2023-24",
            embedding_dimensions=min(32, len(detailed_names)),
            temperature=.07,
            feature_names=detailed_names,
        )
        cohort = features.loc[cohort_mask & features[target].notna()].reset_index(drop=True)
        actual = cohort[target].to_numpy(dtype=float)
        coverage_values = np.concatenate([
            np.sqrt(1 - SIAMESE_DETAIL_WEIGHT) * transform_siamese(siamese, cohort),
            np.sqrt(SIAMESE_DETAIL_WEIGHT) * transform_siamese(detailed, cohort),
        ], axis=1)
        for model, values in (
            ("Denoising Player DNA", transform_style(baseline, cohort, "Denoising autoencoder")),
            ("Siamese encoder", transform_siamese(siamese, cohort)),
            ("Coverage-aware Siamese", coverage_values),
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


def _shortcut_audit(features: pd.DataFrame, artifacts: dict) -> dict:
    """Check excluded metadata, unseen players, and a permuted-identity null."""

    query, candidates = _cohorts(features, "2025-26", "2024-25")
    _, score = _coverage_aware_scores(
        artifacts["siamese"], artifacts["detailed_siamese"],
        query, candidates, SIAMESE_DETAIL_WEIGHT,
    )
    candidate_ids = candidates["player_id"].astype(int).to_numpy()

    def ranks(ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.asarray([
            np.flatnonzero(ids[np.argsort(-score[index])] == int(player_id))[0] + 1
            for index, player_id in enumerate(query["player_id"])
            if mask[index]
        ])

    training_ids = set(features.loc[
        features["offensive_eligible"].fillna(False)
        & features["season"].le("2023-24"),
        "player_id",
    ].astype(int))
    seen = query["player_id"].astype(int).isin(training_ids).to_numpy()
    groups = {}
    for label, mask in (("seen", seen), ("unseen", ~seen)):
        values = ranks(candidate_ids, mask)
        groups[label] = {
            "eligible_queries": len(values),
            "mean_reciprocal_rank": float(np.mean(1 / values)),
            "recall_at_1": float(np.mean(values == 1)),
            "recall_at_5": float(np.mean(values <= 5)),
        }

    rng = np.random.default_rng(73)
    null_mrr = [
        float(np.mean(1 / ranks(rng.permutation(candidate_ids), np.ones(len(query), bool))))
        for _ in range(100)
    ]
    probe = query.head(50).copy()
    before = np.concatenate([
        transform_siamese(artifacts["siamese"], probe),
        transform_siamese(artifacts["detailed_siamese"], probe),
    ], axis=1)
    for column, value in {
        "player_id": -999,
        "player_name": "IDENTITY LEAK",
        "team": "XXX",
        "position": "C",
        "age": 99,
        "height": 999,
        "weight": 999,
        "dpm": 999,
        "o_dpm": 999,
        "d_dpm": 999,
    }.items():
        probe[column] = value
    after = np.concatenate([
        transform_siamese(artifacts["siamese"], probe),
        transform_siamese(artifacts["detailed_siamese"], probe),
    ], axis=1)
    metadata_delta = float(np.max(np.abs(before - after)))
    forbidden = {
        "player_id", "player_name", "team", "position", "height", "weight",
        "age", "dpm", "o_dpm", "d_dpm", "field_goal_percentage",
        "three_point_percentage", "effective_field_goal_percentage",
    }
    return {
        "forbidden_input_features": sorted(forbidden & (
            set(artifacts["siamese"]["feature_names"])
            | set(artifacts["detailed_siamese"]["feature_names"])
        )),
        "metadata_perturbation_max_abs_delta": metadata_delta,
        "era_limited_features_excluded": all(
            name not in artifacts["siamese"]["feature_names"]
            for name in ("floater_frequency", "pull_up_frequency", "step_back_frequency")
        ),
        "latest_test_by_training_exposure": groups,
        "permuted_identity_null_mrr_mean": float(np.mean(null_mrr)),
        "permuted_identity_null_mrr_p95": float(np.quantile(null_mrr, .95)),
    }


def main() -> int:
    root = settings.historical_data_dir
    artifacts = {
        "autoencoder": joblib.load(root / "play_style_models.joblib")["Broad history"]["Offensive"],
        "temporal": joblib.load(
            root / "model_registry/play_style/candidates/temporal_contrastive_v1/model.joblib"
        ),
        "siamese": joblib.load(
            root / "model_registry/play_style/v3/siamese_model.joblib"
        ),
        "detailed_siamese": joblib.load(
            root / "model_registry/play_style/v3/detailed_siamese_model.joblib"
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
    siamese_embeddings = pd.read_parquet(root / "siamese_offensive_embeddings.parquet")
    players = _player_checks(features, embeddings, siamese_embeddings)
    held_out = _held_out_behavior(features)
    shortcut_audit = _shortcut_audit(features, artifacts)
    fold_report = json.loads(
        (ROOT / "models/play_style/siamese_tabular_v3_summary.json").read_text()
    )["holdout_evaluations"]
    fold_frame = pd.DataFrame(fold_report)
    pivot = fold_frame.pivot(index="fold", columns="model")
    fold_deltas = pd.DataFrame({
        metric: pivot[metric]["Coverage-aware ensemble"] - pivot[metric]["Stable Siamese"]
        for metric in ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
    }).reset_index()
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": "siamese-tabular-v3-coverage-aware",
        "production_changed": True,
        "fold_deltas": fold_deltas.to_dict(orient="records"),
        "fold_wins": {
            metric: int((fold_deltas[metric] > 0).sum())
            for metric in ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
        },
        "split_season": split_rows,
        "bootstrap": bootstrap,
        "held_out_behavior": held_out,
        "shortcut_audit": shortcut_audit,
        "representative_player_checks": players,
    }
    local = root / "model_registry/play_style/v3"
    (local / "robustness_evaluation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    bootstrap_rows.to_parquet(
        local / "bootstrap_player_stability.parquet", index=False, compression="snappy"
    )
    tracked = ROOT / "models/play_style/siamese_tabular_v3_robustness.json"
    tracked.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nFold wins\n", payload["fold_wins"])
    print("\nSplit-season means\n", pd.DataFrame(split_rows).groupby("model")[
        ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
    ].mean().to_string())
    print("\nBootstrap\n", pd.DataFrame(bootstrap).to_string(index=False))
    print("\nHeld-out behavior\n", pd.DataFrame(held_out).to_string(index=False))
    for name, result in players.items():
        print(f"\n{name} · {result['season']}")
        print("Coverage-aware:", [row["player_name"] for row in result["coverage_aware"]])
    print(f"Saved {tracked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
