"""Train a non-negative temporal cross-view challenger for Broad History offense."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, normalize


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ball_ai.analytics.play_style import BROAD_OFFENSIVE_FEATURES  # noqa: E402
from ball_ai.config import settings  # noqa: E402
from scripts.evaluate_play_style_rolling import ROLLING_FOLDS  # noqa: E402
from scripts.train_temporal_contrastive_encoder import (  # noqa: E402
    ADDITIONAL_HOLDOUT_FOLDS,
    HOLDOUT_FOLDS,
    _fold_scores,
    _metrics,
)


DEVELOPMENT_FOLDS = ROLLING_FOLDS[1:-1]
LATEST_TEST_FOLD = ROLLING_FOLDS[-1]
TARGET_STRENGTHS = [.25, .50, .75]
COMPONENT_WEIGHTS = [.10, .20, .30]
DEPLOYED_TEMPORAL_WEIGHT = .30


def temporal_targets(
    clean: np.ndarray,
    players: pd.Series,
    seasons: pd.Series,
    strength: float,
) -> np.ndarray:
    """Blend each training row toward its adjacent seasons from the same player."""

    if not 0 <= strength <= 1:
        raise ValueError("strength must be between zero and one")
    targets = clean.copy()
    years = pd.to_numeric(seasons.astype(str).str[:4], errors="coerce").to_numpy()
    groups = pd.Series(np.arange(len(players))).groupby(players.to_numpy()).apply(list)
    for positions in groups:
        positions = np.asarray(positions, dtype=int)
        for position in positions:
            adjacent = positions[np.abs(years[positions] - years[position]) == 1]
            if len(adjacent):
                targets[position] = (
                    (1 - strength) * clean[position]
                    + strength * clean[adjacent].mean(axis=0)
                )
    return targets


def fit_cross_view_encoder(
    features: pd.DataFrame,
    *,
    training_end: str,
    target_strength: float,
    latent_dimensions: int = 16,
    random_state: int = 42,
) -> dict:
    """Fit a denoising encoder whose target includes adjacent-season behavior."""

    training = features.loc[
        features["offensive_eligible"].fillna(False)
        & features["season"].le(training_end)
    ].reset_index(drop=True)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    clean = scaler.fit_transform(imputer.fit_transform(training[BROAD_OFFENSIVE_FEATURES]))
    targets = temporal_targets(
        clean, training["player_id"], training["season"], target_strength
    )
    rng = np.random.default_rng(random_state)
    corrupted = np.tile(clean, (3, 1))
    corrupted += rng.normal(0, .05, size=corrupted.shape)
    corrupted[rng.random(corrupted.shape) < .10] = 0
    encoder = MLPRegressor(
        hidden_layer_sizes=(min(latent_dimensions, clean.shape[1]),),
        activation="tanh",
        alpha=.001,
        early_stopping=True,
        max_iter=500,
        random_state=random_state,
    ).fit(corrupted, np.tile(targets, (3, 1)))
    return {
        "model": "Temporal cross-view autoencoder",
        "feature_names": BROAD_OFFENSIVE_FEATURES,
        "training_end": training_end,
        "training_rows": len(training),
        "target_strength": target_strength,
        "latent_dimensions": min(latent_dimensions, clean.shape[1]),
        "imputer": imputer,
        "scaler": scaler,
        "encoder": encoder,
    }


def transform_cross_view(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    """Transform player-seasons into normalized cross-view embeddings."""

    clean = artifact["scaler"].transform(
        artifact["imputer"].transform(frame[artifact["feature_names"]])
    )
    hidden = np.tanh(
        clean @ artifact["encoder"].coefs_[0] + artifact["encoder"].intercepts_[0]
    )
    return normalize(hidden)


def _cross_view_scores(
    artifact: dict, query: pd.DataFrame, candidates: pd.DataFrame
) -> np.ndarray:
    left = transform_cross_view(artifact, query)
    right = transform_cross_view(artifact, candidates)
    cosine = np.clip(left @ right.T, -1, 1)
    return 1 - np.arccos(cosine) / np.pi


def _summary(rows: pd.DataFrame, model: str) -> dict:
    selected = rows.loc[rows["model"].eq(model)]
    return {
        metric: float(selected[metric].mean())
        for metric in ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
    }


def _evaluate_selected(
    features: pd.DataFrame,
    folds: list[tuple[str, str, str]],
    *,
    target_strength: float,
    component_weight: float,
    label: str,
) -> pd.DataFrame:
    rows = []
    for training_end, query_season, candidate_season in folds:
        print(f"{label}: {query_season} -> {candidate_season}", flush=True)
        query, candidates, hybrid, temporal = _fold_scores(
            features, training_end, query_season, candidate_season
        )
        deployed = (
            (1 - DEPLOYED_TEMPORAL_WEIGHT) * hybrid
            + DEPLOYED_TEMPORAL_WEIGHT * temporal
        )
        if component_weight:
            artifact = fit_cross_view_encoder(
                features,
                training_end=training_end,
                target_strength=target_strength,
            )
            challenger = (
                (1 - component_weight) * deployed
                + component_weight * _cross_view_scores(artifact, query, candidates)
            )
        else:
            challenger = deployed
        for model, scores in (("Frozen v1", deployed), ("Cross-view ensemble", challenger)):
            rows.append({
                "set": label,
                "fold": f"{query_season} -> {candidate_season}",
                "model": model,
                **_metrics(scores, query, candidates),
            })
    return pd.DataFrame(rows)


def main() -> int:
    features = pd.read_parquet(
        settings.historical_data_dir
        / "model_registry/play_style/v1/player_style_features.parquet"
    )
    development_rows = []
    for training_end, query_season, candidate_season in DEVELOPMENT_FOLDS:
        print(f"development: {query_season} -> {candidate_season}", flush=True)
        query, candidates, hybrid, temporal = _fold_scores(
            features, training_end, query_season, candidate_season
        )
        deployed = (
            (1 - DEPLOYED_TEMPORAL_WEIGHT) * hybrid
            + DEPLOYED_TEMPORAL_WEIGHT * temporal
        )
        development_rows.append({
            "fold": f"{query_season} -> {candidate_season}",
            "target_strength": 0,
            "component_weight": 0,
            **_metrics(deployed, query, candidates),
        })
        for strength in TARGET_STRENGTHS:
            artifact = fit_cross_view_encoder(
                features, training_end=training_end, target_strength=strength
            )
            cross_view = _cross_view_scores(artifact, query, candidates)
            for weight in COMPONENT_WEIGHTS:
                development_rows.append({
                    "fold": f"{query_season} -> {candidate_season}",
                    "target_strength": strength,
                    "component_weight": weight,
                    **_metrics((1 - weight) * deployed + weight * cross_view, query, candidates),
                })
    development = pd.DataFrame(development_rows)
    development_summary = (
        development.groupby(["target_strength", "component_weight"], as_index=False)[
            ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
        ]
        .mean()
        .sort_values("mean_reciprocal_rank", ascending=False)
    )
    winner = development_summary.iloc[0]
    strength = float(winner["target_strength"])
    weight = float(winner["component_weight"])
    if weight == 0:
        holdouts = _evaluate_selected(
            features,
            [*HOLDOUT_FOLDS, *ADDITIONAL_HOLDOUT_FOLDS],
            target_strength=TARGET_STRENGTHS[0],
            component_weight=0,
            label="all_19_holdouts",
        )
        strength = 0
    else:
        holdouts = _evaluate_selected(
            features,
            [*HOLDOUT_FOLDS, *ADDITIONAL_HOLDOUT_FOLDS],
            target_strength=strength,
            component_weight=weight,
            label="all_19_holdouts",
        )
    latest = _evaluate_selected(
        features,
        [LATEST_TEST_FOLD],
        target_strength=strength or TARGET_STRENGTHS[0],
        component_weight=weight,
        label="latest_test",
    )
    destination = (
        settings.historical_data_dir
        / "model_registry/play_style/candidates/multiview_temporal_v1"
    )
    destination.mkdir(parents=True, exist_ok=True)
    if weight:
        final_artifact = fit_cross_view_encoder(
            features, training_end="2023-24", target_strength=strength
        )
        joblib.dump(final_artifact, destination / "model.joblib")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": "multiview-temporal-v1",
        "status": "offline_challenger",
        "production_changed": False,
        "objective": (
            "Denoise each player-season while partially reconstructing adjacent-season "
            "behavior; no other player is declared a negative example."
        ),
        "selected_target_strength": strength,
        "selected_component_weight": weight,
        "development_summary": development_summary.to_dict(orient="records"),
        "all_19_holdouts": {
            "frozen_v1": _summary(holdouts, "Frozen v1"),
            "cross_view_ensemble": _summary(holdouts, "Cross-view ensemble"),
        },
        "latest_test": {
            "frozen_v1": _summary(latest, "Frozen v1"),
            "cross_view_ensemble": _summary(latest, "Cross-view ensemble"),
        },
        "holdout_evaluations": holdouts.to_dict(orient="records"),
    }
    (destination / "evaluation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    tracked = ROOT / "models/play_style/multiview_temporal_v1_summary.json"
    tracked.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nDevelopment\n", development_summary.head(10).to_string(index=False))
    print("\nAll 19 holdouts\n", pd.DataFrame(payload["all_19_holdouts"]).to_string())
    print("\nLatest test\n", pd.DataFrame(payload["latest_test"]).to_string())
    print(f"Saved {tracked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
