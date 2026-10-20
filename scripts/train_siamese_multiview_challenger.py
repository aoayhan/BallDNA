"""Test wider, odd/even-view, and reliability-weighted Siamese challengers."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

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
)
from ball_ai.config import settings  # noqa: E402
from scripts.evaluate_broad_v2_robustness import (  # noqa: E402
    _feature_rows,
    _load_games,
    _load_shots,
)
from scripts.evaluate_play_style_rolling import ROLLING_FOLDS  # noqa: E402
from scripts.train_siamese_style_encoder import adjacent_pair_indices  # noqa: E402
from scripts.train_temporal_contrastive_encoder import (  # noqa: E402
    ADDITIONAL_HOLDOUT_FOLDS,
    HOLDOUT_FOLDS,
    _cohorts,
    _metrics,
)


STABLE_FEATURES = list(BROAD_OFFENSIVE_FEATURES)
DETAILED_FEATURES = [*STABLE_FEATURES, *MODERN_MOVE_FEATURES]
DEVELOPMENT_FOLDS = ROLLING_FOLDS[1:-1]
LATEST_TEST_FOLD = ROLLING_FOLDS[-1]
DETAIL_WEIGHT = 0.50
TEMPERATURE = 0.07
EMBEDDING_DIMENSIONS = 32
EPOCHS = 60
BATCH_SIZE = 256
VIEW_CACHE = (
    settings.historical_data_dir
    / "model_registry/play_style/candidates/siamese_multiview_v1/odd_even_features.parquet"
)
VARIANTS = {
    "Frozen v4 reproduction": {"hidden": (64, 32), "views": False, "weighted": False},
    "Wider encoder": {"hidden": (128, 64), "views": False, "weighted": False},
    "Odd/even views": {"hidden": (64, 32), "views": True, "weighted": False},
    "Reliability weighted": {"hidden": (64, 32), "views": False, "weighted": True},
    "Combined challenger": {"hidden": (128, 64), "views": True, "weighted": True},
}


class ChallengerEncoder(nn.Module):
    """Shared MLP used on both sides of each player-style pair."""

    def __init__(
        self, inputs: int, embedding_dimensions: int, hidden: tuple[int, int]
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputs, hidden[0]),
            nn.GELU(),
            nn.Dropout(.10),
            nn.Linear(hidden[0], hidden[1]),
            nn.GELU(),
            nn.Linear(hidden[1], embedding_dimensions),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(values), dim=1)


def weighted_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return supervised contrastive loss with optional anchor reliability weights."""

    similarity = embeddings @ embeddings.T / temperature
    self_mask = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    positive = labels[:, None].eq(labels[None, :]) & ~self_mask
    similarity = similarity.masked_fill(self_mask, -torch.inf)
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1)
    if torch.any(positive_count == 0):
        raise ValueError("Every contrastive anchor needs at least one positive view.")
    anchor_loss = -(
        log_probability.masked_fill(~positive, 0).sum(dim=1) / positive_count
    )
    if weights is None:
        return anchor_loss.mean()
    if len(weights) != len(anchor_loss) or torch.any(weights <= 0):
        raise ValueError("Reliability weights must be positive and match the anchors.")
    return (anchor_loss * weights).sum() / weights.sum()


def alternating_game_views(games: pd.DataFrame) -> pd.DataFrame:
    """Label each player's chronologically ordered games odd or even."""

    ordered = games.sort_values(["player_id", "game_date", "game_id"]).copy()
    number = ordered.groupby("player_id").cumcount() + 1
    ordered["view"] = np.where(number.mod(2).eq(1), "odd", "even")
    return ordered


def _reliability(frame: pd.DataFrame, *, half_season: bool) -> pd.Series:
    scale = 0.5 if half_season else 1.0
    return pd.concat(
        [
            frame["games_played"] / (50 * scale),
            frame["minutes_total"] / (1_500 * scale),
            frame["shot_attempts"] / (500 * scale),
        ],
        axis=1,
    ).min(axis=1).clip(.05, 1)


def build_odd_even_profiles(features: pd.DataFrame) -> pd.DataFrame:
    """Rebuild two disjoint style profiles per training player-season."""

    rows = []
    seasons = sorted(features.loc[features["season"].le("2023-24"), "season"].unique())
    for season in seasons:
        games = alternating_game_views(_load_games(settings.historical_data_dir, season))
        shots = _load_shots(settings.historical_data_dir, season)
        for view in ("odd", "even"):
            profile = _feature_rows(games.loc[games["view"].eq(view)], shots, season)
            profile["view"] = view
            profile["offensive_reliability"] = _reliability(
                profile, half_season=True
            )
            profile = profile.loc[
                profile["games_played"].ge(10)
                & profile["minutes_total"].ge(250)
                & profile["shot_attempts"].ge(50)
            ]
            rows.append(profile)
        print(f"Built odd/even profiles for {season}", flush=True)
    output = pd.concat(rows, ignore_index=True)
    common = output.groupby(["player_id", "season"])["view"].nunique()
    common = common[common.eq(2)].index
    return output.set_index(["player_id", "season"]).loc[common].reset_index()


def _views(features: pd.DataFrame) -> pd.DataFrame:
    if VIEW_CACHE.exists():
        return pd.read_parquet(VIEW_CACHE)
    output = build_odd_even_profiles(features)
    VIEW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(VIEW_CACHE, index=False, compression="snappy")
    return output


def _augment(values: torch.Tensor) -> torch.Tensor:
    noisy = values + torch.randn_like(values) * .03
    return noisy.masked_fill(torch.rand_like(noisy) < .05, 0)


def _pair_rows(full: pd.DataFrame, views: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    adjacent = adjacent_pair_indices(full)
    pairs = [adjacent]
    reliability = [np.sqrt(
        full.iloc[adjacent[:, 0]]["offensive_reliability"].to_numpy(dtype=float)
        * full.iloc[adjacent[:, 1]]["offensive_reliability"].to_numpy(dtype=float)
    )]
    if views is not None:
        indexed = views.reset_index(drop=True).reset_index(names="row")
        paired = indexed.pivot(index=["player_id", "season"], columns="view", values="row")
        paired = paired.dropna(subset=["odd", "even"]).astype(int)
        split_pairs = paired[["odd", "even"]].to_numpy() + len(full)
        pairs.append(split_pairs)
        view_reliability = views["offensive_reliability"].to_numpy(dtype=float)
        reliability.append(np.sqrt(
            view_reliability[split_pairs[:, 0] - len(full)]
            * view_reliability[split_pairs[:, 1] - len(full)]
        ))
    return np.concatenate(pairs), np.concatenate(reliability)


def fit_challenger(
    features: pd.DataFrame,
    views: pd.DataFrame,
    *,
    training_end: str,
    feature_names: list[str],
    hidden: tuple[int, int],
    use_views: bool,
    reliability_weighted: bool,
    training_start: str | None = None,
    random_state: int = 42,
) -> dict:
    """Fit one fold-isolated challenger without changing deployed artifacts."""

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    full = features.loc[
        features["offensive_eligible"].fillna(False)
        & features["season"].le(training_end)
    ].sort_values(["player_id", "season"]).reset_index(drop=True)
    split = views.loc[views["season"].le(training_end)].copy()
    if training_start is not None:
        full = full.loc[full["season"].ge(training_start)].reset_index(drop=True)
        split = split.loc[split["season"].ge(training_start)].reset_index(drop=True)
    split = split.sort_values(["player_id", "season", "view"]).reset_index(drop=True)
    selected_views = split if use_views else None

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputer.fit(full[feature_names])
    scaler.fit(imputer.transform(full[feature_names]))
    training = full if selected_views is None else pd.concat(
        [full, selected_views], ignore_index=True, sort=False
    )
    clean = scaler.transform(imputer.transform(training[feature_names]))
    pairs, pair_reliability = _pair_rows(full, selected_views)
    pair_players = training.iloc[pairs[:, 0]]["player_id"].astype(int).to_numpy()
    values = torch.tensor(clean, dtype=torch.float32)
    model = ChallengerEncoder(len(feature_names), EMBEDDING_DIMENSIONS, hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(random_state)
    model.train()
    final_loss = np.nan
    for _ in range(EPOCHS):
        losses = []
        order = rng.permutation(len(pairs))
        for start in range(0, len(pairs), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            selected = pairs[batch]
            left, right = _augment(values[selected[:, 0]]), _augment(values[selected[:, 1]])
            labels = torch.tensor(pair_players[batch], dtype=torch.long).repeat(2)
            weights = None
            if reliability_weighted:
                weights = torch.tensor(pair_reliability[batch], dtype=torch.float32).repeat(2)
            embeddings = torch.cat([model(left), model(right)], dim=0)
            loss = weighted_supervised_contrastive_loss(
                embeddings, labels, TEMPERATURE, weights
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        final_loss = float(np.mean(losses))
    return {
        "feature_names": feature_names,
        "training_end": training_end,
        "training_start": training_start,
        "hidden": hidden,
        "training_rows": len(training),
        "training_pairs": len(pairs),
        "final_loss": final_loss,
        "imputer": imputer,
        "scaler": scaler,
        "state_dict": model.state_dict(),
    }


def transform(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    clean = artifact["scaler"].transform(
        artifact["imputer"].transform(frame[artifact["feature_names"]])
    )
    model = ChallengerEncoder(
        len(artifact["feature_names"]), EMBEDDING_DIMENSIONS, artifact["hidden"]
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(clean, dtype=torch.float32)).numpy()


def _scores(artifact: dict, query: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
    cosine = np.clip(transform(artifact, query) @ transform(artifact, candidates).T, -1, 1)
    return 1 - np.arccos(cosine) / np.pi


def evaluate_variant(
    features: pd.DataFrame,
    views: pd.DataFrame,
    fold: tuple[str, str, str],
    config: dict,
) -> dict:
    training_end, query_season, candidate_season = fold
    query, candidates = _cohorts(features, query_season, candidate_season)
    stable = fit_challenger(
        features,
        views,
        training_end=training_end,
        feature_names=STABLE_FEATURES,
        hidden=config["hidden"],
        use_views=config["views"],
        reliability_weighted=config["weighted"],
    )
    stable_score = _scores(stable, query, candidates)
    detailed = None
    if training_end >= "2008-09":
        detailed = fit_challenger(
            features,
            views,
            training_start=SIAMESE_DETAIL_START,
            training_end=training_end,
            feature_names=DETAILED_FEATURES,
            hidden=config["hidden"],
            use_views=config["views"],
            reliability_weighted=config["weighted"],
        )
    detailed_score = stable_score if detailed is None else _scores(
        detailed, query, candidates
    )
    coverage = (
        query["season"].ge(SIAMESE_DETAIL_START).to_numpy()[:, None]
        & candidates["season"].ge(SIAMESE_DETAIL_START).to_numpy()[None, :]
    )
    combined = np.where(
        coverage,
        (1 - DETAIL_WEIGHT) * stable_score + DETAIL_WEIGHT * detailed_score,
        stable_score,
    )
    return {
        "fold": f"{query_season} -> {candidate_season}",
        **_metrics(combined, query, candidates),
        "stable_pairs": stable["training_pairs"],
        "detailed_pairs": 0 if detailed is None else detailed["training_pairs"],
    }


def _summary(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby("variant", as_index=False)[
            ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
        ]
        .mean()
        .sort_values(
            ["recall_at_1", "mean_reciprocal_rank", "recall_at_5"],
            ascending=False,
        )
    )


def main() -> int:
    torch.set_num_threads(1)
    features = pd.read_parquet(
        settings.historical_data_dir
        / "model_registry/play_style/v1/player_style_features.parquet"
    )
    views = _views(features)
    development_rows = []
    for variant, config in VARIANTS.items():
        for fold in DEVELOPMENT_FOLDS:
            print(f"Development · {variant} · {fold[1]} -> {fold[2]}", flush=True)
            development_rows.append({
                "variant": variant,
                **evaluate_variant(features, views, fold, config),
            })
    development = pd.DataFrame(development_rows)
    development_summary = _summary(development)
    challenger_summary = development_summary.loc[
        ~development_summary["variant"].eq("Frozen v4 reproduction")
    ]
    winner = challenger_summary.iloc[0]
    selected_variant = str(winner["variant"])
    selected_config = VARIANTS[selected_variant]

    test_rows = []
    for fold in [*HOLDOUT_FOLDS, *ADDITIONAL_HOLDOUT_FOLDS, LATEST_TEST_FOLD]:
        print(f"Locked test · {selected_variant} · {fold[1]} -> {fold[2]}", flush=True)
        test_rows.append({
            "variant": selected_variant,
            "set": "latest_test" if fold == LATEST_TEST_FOLD else "all_19_holdouts",
            **evaluate_variant(features, views, fold, selected_config),
        })
    tests = pd.DataFrame(test_rows)
    test_summary = tests.groupby("set", as_index=False)[
        ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
    ].mean()
    frozen = json.loads(
        (ROOT / "models/play_style/siamese_tabular_v4_summary.json").read_text()
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": "siamese-multiview-v1-challenger",
        "status": "offline_challenger",
        "production_changed": False,
        "objective": (
            "Test a 128 -> 64 shared encoder, disjoint odd/even game views, and "
            "reliability-weighted supervised contrastive InfoNCE."
        ),
        "selected_by": "Development Recall@1, then MRR, then Recall@5",
        "selected_challenger": selected_variant,
        "variants": VARIANTS,
        "view_rows": int(len(views)),
        "development_summary": development_summary.to_dict(orient="records"),
        "development_evaluations": development.to_dict(orient="records"),
        "challenger_test_summary": test_summary.to_dict(orient="records"),
        "challenger_test_evaluations": tests.to_dict(orient="records"),
        "frozen_v4_all_19_holdouts": frozen["all_19_holdouts"]["coverage_aware_ensemble"],
        "frozen_v4_latest_test": frozen["latest_test"]["coverage_aware_ensemble"],
    }
    output = ROOT / "models/play_style/siamese_multiview_v1_summary.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nDevelopment\n", development_summary.to_string(index=False))
    print("\nLocked tests\n", test_summary.to_string(index=False))
    print("\nFrozen v4 all 19\n", payload["frozen_v4_all_19_holdouts"])
    print("\nFrozen v4 latest\n", payload["frozen_v4_latest_test"])
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
