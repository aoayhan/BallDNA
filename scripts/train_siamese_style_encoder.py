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

from ball_ai.analytics.play_style import (  # noqa: E402
    BROAD_OFFENSIVE_FEATURES,
    MODERN_MOVE_FEATURES,
    SIAMESE_DETAIL_START,
    top_siamese_style_pairs,
)
from ball_ai.config import settings  # noqa: E402
from scripts.evaluate_play_style_rolling import ROLLING_FOLDS  # noqa: E402
from scripts.train_temporal_contrastive_encoder import (  # noqa: E402
    ADDITIONAL_HOLDOUT_FOLDS,
    HOLDOUT_FOLDS,
    _cohorts,
    _metrics,
)


DEVELOPMENT_FOLDS = ROLLING_FOLDS[1:-1]
LATEST_TEST_FOLD = ROLLING_FOLDS[-1]
EMBEDDING_DIMENSIONS = [16, 32]
TEMPERATURES = [.07, .15]
DETAIL_WEIGHTS = [.25, .50, .75, 1.0]
MODEL_VERSION = "v4"
STABLE_FEATURES = list(BROAD_OFFENSIVE_FEATURES)
DETAILED_FEATURES = [*STABLE_FEATURES, *MODERN_MOVE_FEATURES]
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
    training_start: str | None = None,
    epochs: int = EPOCHS,
    random_state: int = 42,
) -> dict:
    """Fit one shared encoder using adjacent seasons as self-supervised positives."""

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    columns = list(feature_names or STABLE_FEATURES)
    training = features.loc[
        features["offensive_eligible"].fillna(False)
        & features["season"].le(training_end)
    ].sort_values(["player_id", "season"]).reset_index(drop=True)
    if training_start is not None:
        training = training.loc[training["season"].ge(training_start)].reset_index(drop=True)
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
        "training_start": training_start,
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


def build_siamese_embedding_frame(
    artifact: dict,
    features: pd.DataFrame,
    detailed_artifact: dict | None = None,
) -> pd.DataFrame:
    """Export compact inference vectors so the deployed app does not need PyTorch."""

    eligible = features.loc[features["offensive_eligible"].fillna(False)].copy()
    vectors = transform_siamese(artifact, eligible)
    columns = [f"siamese_embedding_{index:02d}" for index in range(vectors.shape[1])]
    output = pd.concat([
        eligible[["player_id", "season"]].reset_index(drop=True),
        pd.DataFrame(vectors, columns=columns),
    ], axis=1)
    if detailed_artifact is not None:
        detailed = transform_siamese(detailed_artifact, eligible)
        for index in range(detailed.shape[1]):
            output[f"detailed_siamese_embedding_{index:02d}"] = detailed[:, index]
    return output


def export_numpy_inference_artifacts(
    stable: dict,
    detailed: dict,
    destination: Path,
    detail_weight: float,
) -> None:
    """Export production inference arrays without a PyTorch runtime dependency."""

    payload: dict[str, np.ndarray] = {
        "detail_weight": np.asarray(detail_weight),
        "detail_start": np.asarray(SIAMESE_DETAIL_START),
    }
    for prefix, artifact in (("stable", stable), ("detailed", detailed)):
        payload[f"{prefix}_feature_names"] = np.asarray(artifact["feature_names"])
        payload[f"{prefix}_imputer_statistics"] = artifact["imputer"].statistics_
        payload[f"{prefix}_scaler_mean"] = artifact["scaler"].mean_
        payload[f"{prefix}_scaler_scale"] = artifact["scaler"].scale_
        for index, layer in enumerate((0, 3, 5)):
            payload[f"{prefix}_weight_{index}"] = (
                artifact["state_dict"][f"network.{layer}.weight"].detach().cpu().numpy()
            )
            payload[f"{prefix}_bias_{index}"] = (
                artifact["state_dict"][f"network.{layer}.bias"].detach().cpu().numpy()
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)


def _siamese_scores(artifact: dict, query: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
    cosine = np.clip(
        transform_siamese(artifact, query) @ transform_siamese(artifact, candidates).T,
        -1,
        1,
    )
    return 1 - np.arccos(cosine) / np.pi


def _fit_components(
    features: pd.DataFrame,
    training_end: str,
    dimensions: int,
    temperature: float,
) -> tuple[dict, dict | None]:
    stable = fit_siamese_encoder(
        features,
        training_end=training_end,
        embedding_dimensions=dimensions,
        temperature=temperature,
    )
    detailed = None
    if training_end >= "2008-09":
        detailed = fit_siamese_encoder(
            features,
            training_start=SIAMESE_DETAIL_START,
            training_end=training_end,
            embedding_dimensions=dimensions,
            temperature=temperature,
            feature_names=DETAILED_FEATURES,
        )
    return stable, detailed


def _coverage_aware_scores(
    stable: dict,
    detailed: dict | None,
    query: pd.DataFrame,
    candidates: pd.DataFrame,
    detail_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    stable_score = _siamese_scores(stable, query, candidates)
    if detailed is None:
        return stable_score, stable_score
    detailed_score = _siamese_scores(detailed, query, candidates)
    coverage = (
        query["season"].ge(SIAMESE_DETAIL_START).to_numpy()[:, None]
        & candidates["season"].ge(SIAMESE_DETAIL_START).to_numpy()[None, :]
    )
    combined = np.where(
        coverage,
        (1 - detail_weight) * stable_score + detail_weight * detailed_score,
        stable_score,
    )
    return stable_score, combined


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
    detail_weight: float,
    label: str,
) -> pd.DataFrame:
    rows = []
    for training_end, query_season, candidate_season in folds:
        print(f"{label}: {query_season} -> {candidate_season}", flush=True)
        query, candidates = _cohorts(features, query_season, candidate_season)
        stable, detailed = _fit_components(
            features, training_end, dimensions, temperature
        )
        stable_scores, ensemble = _coverage_aware_scores(
            stable, detailed, query, candidates, detail_weight
        )
        for model, scores in (("Stable Siamese", stable_scores), ("Coverage-aware ensemble", ensemble)):
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
        query, candidates = _cohorts(features, query_season, candidate_season)
        for dimensions in EMBEDDING_DIMENSIONS:
            for temperature in TEMPERATURES:
                stable, detailed = _fit_components(
                    features, training_end, dimensions, temperature
                )
                for detail_weight in DETAIL_WEIGHTS:
                    _, ensemble = _coverage_aware_scores(
                        stable, detailed, query, candidates, detail_weight
                    )
                    development_rows.append({
                        "fold": f"{query_season} -> {candidate_season}",
                        "embedding_dimensions": dimensions,
                        "temperature": temperature,
                        "detail_weight": detail_weight,
                        **_metrics(ensemble, query, candidates),
                    })
    development = pd.DataFrame(development_rows)
    development_summary = (
        development.groupby(
            ["embedding_dimensions", "temperature", "detail_weight"], as_index=False
        )[["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]]
        .mean()
        .sort_values("mean_reciprocal_rank", ascending=False)
    )
    winner = development_summary.iloc[0]
    dimensions = int(winner["embedding_dimensions"])
    temperature = float(winner["temperature"])
    detail_weight = float(winner["detail_weight"])
    holdouts = _evaluate_selected(
        features,
        [*HOLDOUT_FOLDS, *ADDITIONAL_HOLDOUT_FOLDS],
        dimensions=dimensions,
        temperature=temperature,
        detail_weight=detail_weight,
        label="all_19_holdouts",
    )
    latest = _evaluate_selected(
        features,
        [LATEST_TEST_FOLD],
        dimensions=dimensions,
        temperature=temperature,
        detail_weight=detail_weight,
        label="latest_test",
    )
    destination = settings.historical_data_dir / f"model_registry/play_style/{MODEL_VERSION}"
    destination.mkdir(parents=True, exist_ok=True)
    stable_artifact, detailed_artifact = _fit_components(
        features, "2023-24", dimensions, temperature
    )
    if detailed_artifact is None:
        raise RuntimeError("Detailed Siamese training requires 2007-08+ history.")
    joblib.dump(stable_artifact, destination / "siamese_model.joblib")
    joblib.dump(detailed_artifact, destination / "detailed_siamese_model.joblib")
    export_numpy_inference_artifacts(
        stable_artifact,
        detailed_artifact,
        ROOT / f"models/play_style/siamese_tabular_{MODEL_VERSION}_inference.npz",
        detail_weight,
    )
    deployed_embeddings = build_siamese_embedding_frame(
        stable_artifact, features, detailed_artifact
    )
    deployed_embeddings.to_parquet(destination / "siamese_offensive_embeddings.parquet", index=False)
    leaderboard_metadata = features[[
        "player_id", "season", "player_name", "position", "shot_attempts",
    ]]
    impact_path = settings.historical_data_dir / "darko_dpm.parquet"
    if impact_path.exists():
        leaderboard_metadata = leaderboard_metadata.merge(
            pd.read_parquet(
                impact_path, columns=["player_id", "season", "dpm", "o_dpm", "d_dpm"]
            ).drop_duplicates(["player_id", "season"], keep="last"),
            on=["player_id", "season"],
            how="left",
            validate="one_to_one",
        )
    else:
        leaderboard_metadata = leaderboard_metadata.merge(
            features[["player_id", "season", "dpm", "o_dpm", "d_dpm"]],
            on=["player_id", "season"],
            how="left",
            validate="one_to_one",
        )
    top_siamese_style_pairs(
        deployed_embeddings,
        leaderboard_metadata,
        top_n=100,
        detail_weight=detail_weight,
    ).to_parquet(
        destination / "style_similarity_leaderboard.parquet", index=False
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": "siamese-tabular-v4-no-duplicate-three-point-rate",
        "status": "promoted_broad_offense_component",
        "production_changed": True,
        "architecture": "shared 64 -> 32 MLPs with 32-dimensional L2-normalized embeddings",
        "objective": "Multi-positive supervised contrastive InfoNCE",
        "positive_pairs": "Adjacent eligible seasons and stochastic feature views",
        "stable_feature_count": len(STABLE_FEATURES),
        "detailed_feature_count": len(DETAILED_FEATURES),
        "detail_coverage_start": SIAMESE_DETAIL_START,
        "input_exclusions": [
            "player identity", "player name", "team", "position", "height", "weight",
            "shooting efficiency", "DARKO impact",
        ],
        "selected_embedding_dimensions": dimensions,
        "selected_temperature": temperature,
        "selected_detail_weight": detail_weight,
        "development_summary": development_summary.to_dict(orient="records"),
        "all_19_holdouts": {
            "stable_siamese": _summary(holdouts, "Stable Siamese"),
            "coverage_aware_ensemble": _summary(holdouts, "Coverage-aware ensemble"),
        },
        "latest_test": {
            "stable_siamese": _summary(latest, "Stable Siamese"),
            "coverage_aware_ensemble": _summary(latest, "Coverage-aware ensemble"),
        },
        "holdout_evaluations": holdouts.to_dict(orient="records"),
    }
    (destination / "evaluation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    tracked = ROOT / f"models/play_style/siamese_tabular_{MODEL_VERSION}_summary.json"
    tracked.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nDevelopment\n", development_summary.head(12).to_string(index=False))
    print("\nAll 19 holdouts\n", pd.DataFrame(payload["all_19_holdouts"]).to_string())
    print("\nLatest test\n", pd.DataFrame(payload["latest_test"]).to_string())
    print(f"Saved {tracked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
