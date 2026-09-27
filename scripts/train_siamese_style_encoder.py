"""Train a Siamese tabular Player DNA challenger with supervised contrastive loss."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


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
EMBEDDING_DIMENSIONS = [16, 32]
TEMPERATURES = [.07, .15]
COMPONENT_WEIGHTS = [.10, .20, .30, .40]
DEPLOYED_TEMPORAL_WEIGHT = .30
EPOCHS = 60
BATCH_SIZE = 256


class TabularEncoder(nn.Module):
    """Small shared encoder for both sides of a player-season pair."""

    def __init__(self, inputs: int, embedding_dimensions: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputs, 64),
            nn.GELU(),
            nn.Dropout(.10),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, embedding_dimensions),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(values), dim=1)


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Pull same-player views together while contrasting other players in the batch."""

    similarity = embeddings @ embeddings.T / temperature
    self_mask = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    positive = labels[:, None].eq(labels[None, :]) & ~self_mask
    similarity = similarity.masked_fill(self_mask, -torch.inf)
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1)
    if torch.any(positive_count == 0):
        raise ValueError("Every contrastive anchor needs at least one positive view.")
    return -(
        log_probability.masked_fill(~positive, 0).sum(dim=1) / positive_count
    ).mean()


def adjacent_pair_indices(frame: pd.DataFrame) -> np.ndarray:
    """Return consecutive-season row pairs belonging to the same player."""

    pairs = []
    years = pd.to_numeric(frame["season"].str[:4], errors="coerce").to_numpy()
    for positions in frame.groupby("player_id", sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        order = positions[np.argsort(years[positions])]
        pairs.extend(
            (int(left), int(right))
            for left, right in zip(order[:-1], order[1:])
            if years[right] - years[left] == 1
        )
    if not pairs:
        raise ValueError("Siamese training needs adjacent eligible player-seasons.")
    return np.asarray(pairs, dtype=int)


def _augment(values: torch.Tensor) -> torch.Tensor:
    noisy = values + torch.randn_like(values) * .03
    return noisy.masked_fill(torch.rand_like(noisy) < .05, 0)


def fit_siamese_encoder(
    features: pd.DataFrame,
    *,
    training_end: str,
    embedding_dimensions: int,
    temperature: float,
    feature_names: list[str] | None = None,
    epochs: int = EPOCHS,
    random_state: int = 42,
) -> dict:
    """Fit one shared encoder using adjacent seasons as self-supervised positives."""

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    columns = list(feature_names or BROAD_OFFENSIVE_FEATURES)
    training = features.loc[
        features["offensive_eligible"].fillna(False)
        & features["season"].le(training_end)
    ].sort_values(["player_id", "season"]).reset_index(drop=True)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    clean = scaler.fit_transform(imputer.fit_transform(training[columns]))
    pairs = adjacent_pair_indices(training)
    values = torch.tensor(clean, dtype=torch.float32)
    pair_players = training.iloc[pairs[:, 0]]["player_id"].astype(int).to_numpy()
    model = TabularEncoder(clean.shape[1], embedding_dimensions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(random_state)
    losses = []
    model.train()
    for _ in range(epochs):
        epoch_losses = []
        order = rng.permutation(len(pairs))
        for start in range(0, len(pairs), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            selected = pairs[batch]
            left = _augment(values[selected[:, 0]])
            right = _augment(values[selected[:, 1]])
            labels = torch.tensor(pair_players[batch], dtype=torch.long).repeat(2)
            embeddings = torch.cat([model(left), model(right)], dim=0)
            loss = supervised_contrastive_loss(embeddings, labels, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
    return {
        "model": "Siamese tabular encoder",
        "feature_names": columns,
        "training_end": training_end,
        "training_rows": len(training),
        "training_pairs": len(pairs),
        "embedding_dimensions": embedding_dimensions,
        "temperature": temperature,
        "epochs": epochs,
        "final_loss": losses[-1],
        "imputer": imputer,
        "scaler": scaler,
        "state_dict": model.state_dict(),
    }


def transform_siamese(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    """Encode player-seasons with a fitted Siamese artifact."""

    clean = artifact["scaler"].transform(
        artifact["imputer"].transform(frame[artifact["feature_names"]])
    )
    model = TabularEncoder(len(artifact["feature_names"]), artifact["embedding_dimensions"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(clean, dtype=torch.float32)).numpy()


def build_siamese_embedding_frame(artifact: dict, features: pd.DataFrame) -> pd.DataFrame:
    """Export compact inference vectors so the deployed app does not need PyTorch."""

    eligible = features.loc[features["offensive_eligible"].fillna(False)].copy()
    vectors = transform_siamese(artifact, eligible)
    columns = [f"siamese_embedding_{index:02d}" for index in range(vectors.shape[1])]
    return pd.concat([
        eligible[["player_id", "season"]].reset_index(drop=True),
        pd.DataFrame(vectors, columns=columns),
    ], axis=1)


def _siamese_scores(artifact: dict, query: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
    cosine = np.clip(
        transform_siamese(artifact, query) @ transform_siamese(artifact, candidates).T,
        -1,
        1,
    )
    return 1 - np.arccos(cosine) / np.pi


def _deployed_scores(hybrid: np.ndarray, temporal: np.ndarray) -> np.ndarray:
    return (1 - DEPLOYED_TEMPORAL_WEIGHT) * hybrid + DEPLOYED_TEMPORAL_WEIGHT * temporal


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
    dimensions: int,
    temperature: float,
    component_weight: float,
    label: str,
) -> pd.DataFrame:
    rows = []
    for training_end, query_season, candidate_season in folds:
        print(f"{label}: {query_season} -> {candidate_season}", flush=True)
        query, candidates, hybrid, temporal = _fold_scores(
            features, training_end, query_season, candidate_season
        )
        deployed = _deployed_scores(hybrid, temporal)
        if component_weight:
            artifact = fit_siamese_encoder(
                features,
                training_end=training_end,
                embedding_dimensions=dimensions,
                temperature=temperature,
            )
            siamese = _siamese_scores(artifact, query, candidates)
            challenger = (1 - component_weight) * deployed + component_weight * siamese
        else:
            challenger = deployed
        for model, scores in (("Frozen v1", deployed), ("Siamese ensemble", challenger)):
            rows.append({
                "set": label,
                "fold": f"{query_season} -> {candidate_season}",
                "model": model,
                **_metrics(scores, query, candidates),
            })
    return pd.DataFrame(rows)


def main() -> int:
    torch.set_num_threads(1)
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
        deployed = _deployed_scores(hybrid, temporal)
        development_rows.append({
            "fold": f"{query_season} -> {candidate_season}",
            "embedding_dimensions": 0,
            "temperature": 0,
            "component_weight": 0,
            **_metrics(deployed, query, candidates),
        })
        for dimensions in EMBEDDING_DIMENSIONS:
            for temperature in TEMPERATURES:
                artifact = fit_siamese_encoder(
                    features,
                    training_end=training_end,
                    embedding_dimensions=dimensions,
                    temperature=temperature,
                )
                siamese = _siamese_scores(artifact, query, candidates)
                for weight in COMPONENT_WEIGHTS:
                    development_rows.append({
                        "fold": f"{query_season} -> {candidate_season}",
                        "embedding_dimensions": dimensions,
                        "temperature": temperature,
                        "component_weight": weight,
                        **_metrics((1 - weight) * deployed + weight * siamese, query, candidates),
                    })
    development = pd.DataFrame(development_rows)
    development_summary = (
        development.groupby(
            ["embedding_dimensions", "temperature", "component_weight"], as_index=False
        )[["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]]
        .mean()
        .sort_values("mean_reciprocal_rank", ascending=False)
    )
    winner = development_summary.iloc[0]
    dimensions = int(winner["embedding_dimensions"])
    temperature = float(winner["temperature"])
    weight = float(winner["component_weight"])
    holdouts = _evaluate_selected(
        features,
        [*HOLDOUT_FOLDS, *ADDITIONAL_HOLDOUT_FOLDS],
        dimensions=dimensions or EMBEDDING_DIMENSIONS[0],
        temperature=temperature or TEMPERATURES[0],
        component_weight=weight,
        label="all_19_holdouts",
    )
    latest = _evaluate_selected(
        features,
        [LATEST_TEST_FOLD],
        dimensions=dimensions or EMBEDDING_DIMENSIONS[0],
        temperature=temperature or TEMPERATURES[0],
        component_weight=weight,
        label="latest_test",
    )
    destination = (
        settings.historical_data_dir
        / "model_registry/play_style/candidates/siamese_v1"
    )
    destination.mkdir(parents=True, exist_ok=True)
    if weight:
        artifact = fit_siamese_encoder(
            features,
            training_end="2023-24",
            embedding_dimensions=dimensions,
            temperature=temperature,
        )
        joblib.dump(artifact, destination / "model.joblib")
        build_siamese_embedding_frame(artifact, features).to_parquet(
            settings.historical_data_dir / "siamese_offensive_embeddings.parquet",
            index=False,
        )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": "siamese-tabular-v1",
        "status": "promoted_broad_offense_component",
        "production_changed": True,
        "architecture": "19 -> 64 -> 32 -> L2-normalized embedding",
        "objective": "Multi-positive supervised contrastive InfoNCE",
        "positive_pairs": "Adjacent eligible seasons and stochastic feature views",
        "input_exclusions": [
            "player identity", "player name", "team", "position", "height", "weight",
            "shooting efficiency", "DARKO impact",
        ],
        "selected_embedding_dimensions": dimensions,
        "selected_temperature": temperature,
        "selected_component_weight": weight,
        "development_summary": development_summary.to_dict(orient="records"),
        "all_19_holdouts": {
            "frozen_v1": _summary(holdouts, "Frozen v1"),
            "siamese_ensemble": _summary(holdouts, "Siamese ensemble"),
        },
        "latest_test": {
            "frozen_v1": _summary(latest, "Frozen v1"),
            "siamese_ensemble": _summary(latest, "Siamese ensemble"),
        },
        "holdout_evaluations": holdouts.to_dict(orient="records"),
    }
    (destination / "evaluation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    tracked = ROOT / "models/play_style/siamese_tabular_v1_summary.json"
    tracked.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nDevelopment\n", development_summary.head(12).to_string(index=False))
    print("\nAll 19 holdouts\n", pd.DataFrame(payload["all_19_holdouts"]).to_string())
    print("\nLatest test\n", pd.DataFrame(payload["latest_test"]).to_string())
    print(f"Saved {tracked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
