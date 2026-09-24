"""Train and evaluate the Broad History temporal metric challenger."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ball_ai.analytics.play_style import (  # noqa: E402
    ABSENCE_THRESHOLD,
    BROAD_OFFENSIVE_FEATURES,
    OFFENSIVE_PRESENCE_WEIGHT,
    SHARED_ABSENCE_SHARE,
    TEMPORAL_ENSEMBLE_WEIGHT,
    _absence_information,
    build_embedding_frame,
    fit_style_artifact,
    fit_temporal_contrastive_artifact,
    transform_temporal_contrastive,
)
from ball_ai.config import settings  # noqa: E402
from evaluate_play_style_rolling import ROLLING_FOLDS  # noqa: E402


DEVELOPMENT_FOLDS = ROLLING_FOLDS[1:-1]
HOLDOUT_FOLDS = [
    ("1997-98", "1999-00", "1998-99"),
    ("2003-04", "2005-06", "2004-05"),
    ("2009-10", "2011-12", "2010-11"),
    ("2015-16", "2017-18", "2016-17"),
    ("2019-20", "2021-22", "2020-21"),
]
# Query seasons not used for temporal-weight selection or the first holdout report.
ADDITIONAL_HOLDOUT_FOLDS = [
    ("1998-99", "2000-01", "1999-00"),
    ("1999-00", "2001-02", "2000-01"),
    ("2000-01", "2002-03", "2001-02"),
    ("2002-03", "2004-05", "2003-04"),
    ("2004-05", "2006-07", "2005-06"),
    ("2005-06", "2007-08", "2006-07"),
    ("2007-08", "2009-10", "2008-09"),
    ("2008-09", "2010-11", "2009-10"),
    ("2010-11", "2012-13", "2011-12"),
    ("2012-13", "2014-15", "2013-14"),
    ("2013-14", "2015-16", "2014-15"),
    ("2014-15", "2016-17", "2015-16"),
    ("2017-18", "2019-20", "2018-19"),
    ("2018-19", "2020-21", "2019-20"),
]
ENSEMBLE_CANDIDATES = np.arange(0, 0.51, 0.05)


def _cohorts(
    features: pd.DataFrame, query_season: str, candidate_season: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = features["offensive_eligible"].fillna(False)
    query = features.loc[eligible & features["season"].eq(query_season)].copy()
    candidates = features.loc[
        eligible & features["season"].eq(candidate_season)
    ].copy()
    shared = set(query["player_id"].astype(int)) & set(candidates["player_id"].astype(int))
    return (
        query.loc[query["player_id"].astype(int).isin(shared)].reset_index(drop=True),
        candidates.reset_index(drop=True),
    )


def _metrics(score: np.ndarray, query: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    candidate_ids = candidates["player_id"].astype(int).to_numpy()
    ranks = np.asarray([
        np.flatnonzero(candidate_ids[np.argsort(-score[index])] == int(player_id))[0] + 1
        for index, player_id in enumerate(query["player_id"])
    ])
    return {
        "eligible_queries": len(ranks),
        "mean_reciprocal_rank": float(np.mean(1 / ranks)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
    }


def _hybrid_scores(
    features: pd.DataFrame,
    artifact: dict,
    embeddings: pd.DataFrame,
    query: pd.DataFrame,
    candidates: pd.DataFrame,
) -> np.ndarray:
    columns = [column for column in embeddings if column.startswith("embedding_")]
    indexed_embeddings = embeddings.set_index(["player_id", "season"])
    query_keys = pd.MultiIndex.from_frame(query[["player_id", "season"]])
    candidate_keys = pd.MultiIndex.from_frame(candidates[["player_id", "season"]])
    query_embedding = indexed_embeddings.reindex(query_keys)[columns].to_numpy()
    candidate_embedding = indexed_embeddings.reindex(candidate_keys)[columns].to_numpy()
    cosine = np.clip(query_embedding @ candidate_embedding.T, -1, 1)
    embedding_score = 1 - np.arccos(cosine) / np.pi

    names = artifact["feature_names"]
    raw = artifact["imputer"].transform(features[names])
    scale = np.where(artifact["scaler"].scale_ == 0, 1, artifact["scaler"].scale_)
    positive = np.clip(raw / scale * artifact["feature_weights"], 0, None)
    lookup = {
        (int(player_id), season): (positive_row, raw_row)
        for player_id, season, positive_row, raw_row in zip(
            features["player_id"], features["season"], positive, raw
        )
    }
    query_positive = np.stack([
        lookup[(int(row.player_id), row.season)][0] for row in query.itertuples()
    ])
    candidate_positive = np.stack([
        lookup[(int(row.player_id), row.season)][0] for row in candidates.itertuples()
    ])
    positive_overlap = np.minimum(
        query_positive[:, None], candidate_positive
    ).sum(axis=2) / np.maximum(query_positive[:, None], candidate_positive).sum(axis=2)

    sparse = np.asarray([
        name.endswith("_frequency")
        or name in {"three_point_attempt_rate", "free_throw_attempt_rate"}
        for name in names
    ])
    query_raw = np.stack([
        lookup[(int(row.player_id), row.season)][1] for row in query.itertuples()
    ])[:, sparse]
    candidate_raw = np.stack([
        lookup[(int(row.player_id), row.season)][1] for row in candidates.itertuples()
    ])[:, sparse]
    information = _absence_information(features, artifact)[sparse]
    query_absent = query_raw <= ABSENCE_THRESHOLD
    candidate_absent = candidate_raw <= ABSENCE_THRESHOLD
    shared_absence = (
        (query_absent[:, None] & candidate_absent) * information
    ).sum(axis=2)
    either_absent = (
        (query_absent[:, None] | candidate_absent) * information
    ).sum(axis=2)
    absence_overlap = np.divide(
        shared_absence,
        either_absent,
        out=np.zeros_like(shared_absence),
        where=either_absent != 0,
    )
    behavior = (
        (1 - SHARED_ABSENCE_SHARE) * positive_overlap
        + SHARED_ABSENCE_SHARE * absence_overlap
    )
    return (
        (1 - OFFENSIVE_PRESENCE_WEIGHT) * embedding_score
        + OFFENSIVE_PRESENCE_WEIGHT * behavior
    )


def _fold_scores(
    features: pd.DataFrame,
    training_end: str,
    query_season: str,
    candidate_season: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    query, candidates = _cohorts(features, query_season, candidate_season)
    autoencoder = fit_style_artifact(
        features,
        "Offensive",
        profile="Broad history",
        training_end=training_end,
        latent_dimensions=16,
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
    return query, candidates, hybrid, temporal_score


def _evaluate_folds(
    features: pd.DataFrame,
    folds: list[tuple[str, str, str]],
    temporal_weight: float,
    label: str,
) -> pd.DataFrame:
    rows = []
    for training_end, query_season, candidate_season in folds:
        print(f"{label} fold {query_season} -> {candidate_season}")
        query, candidates, hybrid, temporal = _fold_scores(
            features, training_end, query_season, candidate_season
        )
        rows.extend([
            {
                "set": label,
                "fold": f"{query_season} -> {candidate_season}",
                "model": "Current hybrid",
                **_metrics(hybrid, query, candidates),
            },
            {
                "set": label,
                "fold": f"{query_season} -> {candidate_season}",
                "model": "Temporal ensemble",
                **_metrics(
                    (1 - temporal_weight) * hybrid + temporal_weight * temporal,
                    query,
                    candidates,
                ),
            },
        ])
    return pd.DataFrame(rows)


def main() -> int:
    root = settings.historical_data_dir
    features = pd.read_parquet(
        root / "model_registry/play_style/v1/player_style_features.parquet"
    )
    development_rows = []
    for training_end, query_season, candidate_season in DEVELOPMENT_FOLDS:
        print(f"Development fold {query_season} -> {candidate_season}")
        query, candidates, hybrid, temporal = _fold_scores(
            features, training_end, query_season, candidate_season
        )
        for weight in ENSEMBLE_CANDIDATES:
            development_rows.append({
                "fold": f"{query_season} -> {candidate_season}",
                "temporal_weight": round(float(weight), 2),
                **_metrics((1 - weight) * hybrid + weight * temporal, query, candidates),
            })
    development = pd.DataFrame(development_rows)
    selected = (
        development.groupby("temporal_weight", as_index=False)[
            ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
        ]
        .mean()
        .sort_values("mean_reciprocal_rank", ascending=False)
        .iloc[0]
    )
    selected_weight = float(selected["temporal_weight"])

    holdout = _evaluate_folds(features, HOLDOUT_FOLDS, selected_weight, "original_holdout")
    additional_holdout = _evaluate_folds(
        features,
        ADDITIONAL_HOLDOUT_FOLDS,
        selected_weight,
        "additional_holdout",
    )
    established_rolling = _evaluate_folds(
        features,
        ROLLING_FOLDS[1:],
        selected_weight,
        "established_rolling",
    )
    all_holdouts = pd.concat([holdout, additional_holdout], ignore_index=True)
    holdout_summary = (
        holdout.groupby("model", as_index=False)[
            ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
        ].mean()
    )
    additional_holdout_summary = (
        additional_holdout.groupby("model", as_index=False)[
            ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
        ].mean()
    )
    combined_holdout_summary = (
        all_holdouts.groupby("model", as_index=False)[
            ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
        ].mean()
    )
    established_rolling_summary = (
        established_rolling.groupby("model", as_index=False)[
            ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
        ].mean()
    )
    paired = all_holdouts.pivot(index=["set", "fold"], columns="model")
    fold_deltas = pd.DataFrame({
        metric: paired[metric]["Temporal ensemble"] - paired[metric]["Current hybrid"]
        for metric in ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
    }).reset_index()

    destination = root / "model_registry/play_style/candidates/temporal_contrastive_v1"
    destination.mkdir(parents=True, exist_ok=True)
    final_artifact = fit_temporal_contrastive_artifact(
        features, training_end="2023-24", latent_dimensions=16
    )
    joblib.dump(final_artifact, destination / "model.joblib")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": "temporal-contrastive-v1",
        "status": "promoted_component",
        "objective": (
            "Neighborhood Components Analysis learns a projection where repeated seasons "
            "from one player are positives and other player-seasons are negatives."
        ),
        "input_exclusions": [
            "player identity", "player name", "team", "position", "height", "weight",
            "shooting efficiency", "DARKO impact",
        ],
        "selected_temporal_weight": selected_weight,
        "selected_dimensions": 16,
        "development_summary": selected.to_dict(),
        "holdout_folds": [
            {"training_end": train, "query_season": query, "candidate_season": candidate}
            for train, query, candidate in HOLDOUT_FOLDS
        ],
        "holdout_summary": holdout_summary.to_dict(orient="records"),
        "holdout_evaluations": holdout.to_dict(orient="records"),
        "additional_holdout_folds": [
            {"training_end": train, "query_season": query, "candidate_season": candidate}
            for train, query, candidate in ADDITIONAL_HOLDOUT_FOLDS
        ],
        "additional_holdout_summary": additional_holdout_summary.to_dict(orient="records"),
        "additional_holdout_evaluations": additional_holdout.to_dict(orient="records"),
        "combined_holdout_summary": combined_holdout_summary.to_dict(orient="records"),
        "established_rolling_note": (
            "The 1998-99 -> 1997-98 fold is excluded because only one prior season exists, "
            "so temporal metric learning has no repeated-player training pairs."
        ),
        "established_rolling_summary": established_rolling_summary.to_dict(orient="records"),
        "established_rolling_evaluations": established_rolling.to_dict(orient="records"),
        "fold_deltas": fold_deltas.to_dict(orient="records"),
        "deployment": {
            "scope": "Broad history / Offensive / Denoising autoencoder",
            "weights": {
                "denoising_player_dna": 0.42,
                "temporal_metric": 0.30,
                "positive_behavior_overlap": 0.266,
                "information_weighted_shared_absence": 0.014,
            },
            "rollback_weights": {
                "denoising_player_dna": 0.60,
                "temporal_metric": 0.0,
                "positive_behavior_overlap": 0.38,
                "information_weighted_shared_absence": 0.02,
            },
        },
    }
    (destination / "evaluation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    summary_path = ROOT / "models/play_style/temporal_contrastive_v1_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(holdout_summary.to_string(index=False))
    print("\nAdditional holdout")
    print(additional_holdout_summary.to_string(index=False))
    print("\nCombined holdout")
    print(combined_holdout_summary.to_string(index=False))
    print("\nEstablished rolling")
    print(established_rolling_summary.to_string(index=False))
    print("\nFold deltas (ensemble - hybrid)")
    print(fold_deltas.to_string(index=False))
    print(f"Saved {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
