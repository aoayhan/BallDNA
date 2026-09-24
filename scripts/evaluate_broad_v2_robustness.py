"""Stress-test Broad History v2 without changing the live application model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ball_ai.analytics.play_style import (  # noqa: E402
    build_embedding_frame,
    evaluate_temporal_retrieval,
    find_style_neighbors,
    fit_style_artifact,
    transform_style,
)
from ball_ai.analytics.shot_profile import (  # noqa: E402
    BROAD_V2_SHOT_FEATURES,
    build_broad_v2_shot_profiles,
    build_shot_profiles,
)
from ball_ai.config import settings  # noqa: E402


SPLIT_SEASONS = [
    "1997-98", "2002-03", "2007-08", "2012-13", "2017-18",
    "2022-23", "2024-25", "2025-26",
]
BOOTSTRAP_SEASON = "2024-25"
METHODS = ["Cosine baseline", "PCA baseline", "Denoising autoencoder"]
LATERAL_FEATURES = {
    "left_side_frequency", "center_channel_frequency", "right_side_frequency",
    "lateral_location_std",
}
FINE_DISTANCE_FEATURES = {
    feature for feature in BROAD_V2_SHOT_FEATURES if feature.startswith("distance_")
} | {"shot_distance_std"}
ACTION_LABEL_FEATURES = {
    "dunk_frequency", "layup_frequency", "floater_frequency", "hook_frequency",
    "pull_up_frequency", "step_back_frequency",
}
HOLDOUT_TARGETS = {
    # Remove complete compositional families: retaining all complementary shares
    # would let kNN reconstruct the held-out value arithmetically.
    "rim_frequency": {
        "rim_frequency", "paint_frequency", "midrange_frequency", "three_point_frequency",
        "restricted_area_frequency", "non_restricted_paint_frequency",
        *FINE_DISTANCE_FEATURES,
    },
    "corner_three_frequency": {
        "corner_three_frequency", "above_break_three_frequency", "three_point_frequency",
    },
    "free_throw_attempt_rate": {"free_throw_attempt_rate"},
    "assists_per_36": {"assists_per_36", "assist_turnover_ratio"},
    "offensive_rebounds_per_36": {"offensive_rebounds_per_36"},
    "average_shot_distance": {
        "average_shot_distance", "shot_distance_std", *FINE_DISTANCE_FEATURES,
        "rim_frequency", "paint_frequency", "midrange_frequency", "three_point_frequency",
        "restricted_area_frequency", "non_restricted_paint_frequency",
        "corner_three_frequency", "above_break_three_frequency",
    },
}


def _arguments() -> argparse.Namespace:
    candidate = (
        settings.historical_data_dir / "model_registry" / "play_style"
        / "candidates" / "broad_v2"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=candidate)
    parser.add_argument("--bootstrap-repeats", type=int, default=20)
    return parser.parse_args()


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    top = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    bottom = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    return pd.Series(
        np.divide(top, bottom, out=np.full(len(top), np.nan), where=bottom != 0),
        index=numerator.index,
    )


def _load_games(root: Path, season: str) -> pd.DataFrame:
    paths = sorted(
        (root / "player_game_stats" / f"season={season}" / "season_type=regular")
        .glob("*.parquet")
    )
    if not paths:
        raise FileNotFoundError(f"No regular-season game rows are available for {season}.")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    frame["game_id"] = pd.to_numeric(frame["game_id"], errors="coerce").astype("Int64")
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    return frame.dropna(subset=["player_id", "game_id", "game_date"])


def _load_shots(root: Path, season: str) -> pd.DataFrame:
    year = int(season[:4])
    columns = [
        "PLAYER_ID", "GAME_ID", "SHOT_MADE_FLAG", "SHOT_DISTANCE", "ACTION_TYPE",
        "SHOT_ZONE_BASIC", "SHOT_TYPE", "SHOT_ZONE_AREA", "LOC_X",
    ]
    frame = pd.read_parquet(
        root / "shot_events.parquet",
        columns=columns,
        filters=[[('_season', '=', year), ('_season_type', '=', 'rg')]],
    ).rename(columns={
        "PLAYER_ID": "player_id", "GAME_ID": "game_id",
        "SHOT_MADE_FLAG": "shot_made", "SHOT_DISTANCE": "shot_distance",
        "ACTION_TYPE": "action_type", "SHOT_ZONE_BASIC": "shot_zone_basic",
        "SHOT_TYPE": "shot_type", "SHOT_ZONE_AREA": "shot_zone_area", "LOC_X": "loc_x",
    })
    frame["game_id"] = pd.to_numeric(frame["game_id"], errors="coerce").astype("Int64")
    return frame.dropna(subset=["player_id", "game_id"])


def _feature_rows(games: pd.DataFrame, shots: pd.DataFrame, season: str) -> pd.DataFrame:
    """Rebuild the candidate inputs from an arbitrary set of player-games."""

    totals = {
        "minutes": "minutes_total", "field_goal_attempts": "field_goal_attempts_total",
        "three_point_attempts": "three_point_attempts_total",
        "free_throw_attempts": "free_throw_attempts_total",
        "offensive_rebounds": "offensive_rebounds_total", "assists": "assists_total",
        "turnovers": "turnovers_total",
    }
    grouped = games.groupby("player_id", sort=False)
    frame = grouped[list(totals)].sum(min_count=1).rename(columns=totals).reset_index()
    frame["games_played"] = grouped.size().to_numpy()
    names = grouped["player_name"].first()
    frame["player_name"] = frame["player_id"].map(names)
    frame["season"] = season
    for target, source in {
        "field_goal_attempts_per_36": "field_goal_attempts_total",
        "assists_per_36": "assists_total", "turnovers_per_36": "turnovers_total",
        "offensive_rebounds_per_36": "offensive_rebounds_total",
    }.items():
        frame[target] = 36 * _safe_divide(frame[source], frame["minutes_total"])
    frame["three_point_attempt_rate"] = _safe_divide(
        frame["three_point_attempts_total"], frame["field_goal_attempts_total"]
    )
    frame["free_throw_attempt_rate"] = _safe_divide(
        frame["free_throw_attempts_total"], frame["field_goal_attempts_total"]
    )
    frame["assist_turnover_ratio"] = _safe_divide(
        frame["assists_total"], frame["turnovers_total"]
    )
    used = (
        frame["field_goal_attempts_total"] + 0.44 * frame["free_throw_attempts_total"]
        + frame["turnovers_total"]
    )
    frame["estimated_used_possessions_per_36"] = 36 * _safe_divide(
        used, frame["minutes_total"]
    )

    keys = games[["player_id", "game_id"]]
    sampled_shots = shots.merge(keys, on=["player_id", "game_id"], how="inner")
    profiles = build_shot_profiles(sampled_shots).merge(
        build_broad_v2_shot_profiles(sampled_shots), on="player_id", how="inner",
        validate="one_to_one",
    )
    return frame.merge(profiles, on="player_id", how="inner", validate="one_to_one")


def _eligible(frame: pd.DataFrame, *, half: bool = False) -> pd.DataFrame:
    games = 10 if half else 20
    minutes = 250 if half else 500
    shots = 50 if half else 100
    return frame.loc[
        frame["games_played"].ge(games)
        & frame["minutes_total"].ge(minutes)
        & frame["shot_attempts"].ge(shots)
    ].reset_index(drop=True)


def _split_season(root: Path, season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = _load_games(root, season).sort_values(["player_id", "game_date", "game_id"])
    order = games.groupby("player_id").cumcount()
    count = games.groupby("player_id")["player_id"].transform("size")
    games = games.assign(_half=np.where(order < count / 2, "first", "second"))
    shots = _load_shots(root, season)
    return tuple(
        _eligible(_feature_rows(games.loc[games["_half"].eq(half)], shots, season), half=True)
        for half in ("first", "second")
    )


def retrieval_metrics(
    query_ids: np.ndarray,
    query_values: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_values: np.ndarray,
) -> dict:
    """Score same-player retrieval between two independently observed samples."""

    shared = np.intersect1d(query_ids, candidate_ids)
    query_mask = np.isin(query_ids, shared)
    candidate_mask = np.isin(candidate_ids, shared)
    query_ids = query_ids[query_mask]
    query_values = query_values[query_mask]
    candidate_ids = candidate_ids[candidate_mask]
    candidate_values = candidate_values[candidate_mask]
    scores = query_values @ candidate_values.T
    ranks, own_scores, best_other = [], [], []
    for index, player_id in enumerate(query_ids):
        order = np.argsort(-scores[index])
        target = int(np.flatnonzero(candidate_ids[order] == player_id)[0])
        ranks.append(target + 1)
        own_scores.append(float(scores[index, order[target]]))
        best_other.append(float(scores[index, order[candidate_ids[order] != player_id][0]]))
    ranks = np.asarray(ranks)
    return {
        "eligible_queries": int(len(ranks)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_3": float(np.mean(ranks <= 3)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mean_reciprocal_rank": float(np.mean(1 / ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_self_cosine": float(np.mean(own_scores)),
        "mean_best_other_cosine": float(np.mean(best_other)),
        "mean_self_margin": float(np.mean(np.asarray(own_scores) - best_other)),
    }


def _split_metrics(
    first: pd.DataFrame, second: pd.DataFrame, artifact: dict, method: str,
) -> dict:
    return retrieval_metrics(
        first["player_id"].to_numpy(), transform_style(artifact, first, method),
        second["player_id"].to_numpy(), transform_style(artifact, second, method),
    )


def neighbor_overlap(reference: np.ndarray, sampled: np.ndarray, top_n: int = 10) -> np.ndarray:
    """Return per-row Jaccard overlap between two same-cohort neighbor lists."""

    if len(reference) <= top_n:
        raise ValueError("The cohort must contain more rows than the requested neighbors.")
    first = reference @ reference.T
    second = sampled @ sampled.T
    np.fill_diagonal(first, -np.inf)
    np.fill_diagonal(second, -np.inf)
    first_neighbors = np.argsort(-first, axis=1)[:, :top_n]
    second_neighbors = np.argsort(-second, axis=1)[:, :top_n]
    return np.asarray([
        len(set(left) & set(right)) / len(set(left) | set(right))
        for left, right in zip(first_neighbors, second_neighbors)
    ])


def _bootstrap(
    root: Path, season: str, artifact: dict, repeats: int,
) -> tuple[dict, pd.DataFrame]:
    games, shots = _load_games(root, season), _load_shots(root, season)
    full = _eligible(_feature_rows(games, shots, season))
    full_ids = full["player_id"].to_numpy()
    attempts = full.set_index("player_id")["shot_attempts"]
    player_rows = []
    for repeat in range(repeats):
        sampled_games = (
            games.groupby("player_id", group_keys=False)
            .sample(frac=1, replace=True, random_state=42 + repeat)
        )
        sampled = _eligible(_feature_rows(sampled_games, shots, season))
        common = np.intersect1d(full_ids, sampled["player_id"].to_numpy())
        full_order = full.set_index("player_id").loc[common].reset_index()
        sample_order = sampled.set_index("player_id").loc[common].reset_index()
        base = transform_style(artifact, full_order, "Denoising autoencoder")
        boot = transform_style(artifact, sample_order, "Denoising autoencoder")
        overlap = neighbor_overlap(base, boot)
        cosine = np.sum(base * boot, axis=1)
        player_rows.extend({
            "repeat": repeat, "player_id": int(player_id),
            "player_name": name, "embedding_cosine": float(similarity),
            "top_10_neighbor_jaccard": float(jaccard),
            "shot_attempts": int(attempts.loc[player_id]),
        } for player_id, name, similarity, jaccard in zip(
            common, full_order["player_name"], cosine, overlap
        ))
        print(f"Bootstrap repeat {repeat + 1}/{repeats}")
    rows = pd.DataFrame(player_rows)
    rows["shot_attempt_bin"] = pd.cut(
        rows["shot_attempts"], [0, 199, 399, 799, np.inf],
        labels=["100-199", "200-399", "400-799", "800+"],
    )
    by_attempts = rows.groupby("shot_attempt_bin", observed=True).agg(
        observations=("player_id", "size"),
        mean_embedding_cosine=("embedding_cosine", "mean"),
        mean_top_10_neighbor_jaccard=("top_10_neighbor_jaccard", "mean"),
    ).reset_index()
    summary = {
        "season": season, "repeats": repeats, "eligible_players": int(len(full)),
        "mean_embedding_cosine": float(rows["embedding_cosine"].mean()),
        "p10_embedding_cosine": float(rows["embedding_cosine"].quantile(.10)),
        "mean_top_10_neighbor_jaccard": float(rows["top_10_neighbor_jaccard"].mean()),
        "p10_top_10_neighbor_jaccard": float(rows["top_10_neighbor_jaccard"].quantile(.10)),
        "stability_by_shot_attempts": by_attempts.to_dict(orient="records"),
    }
    return summary, rows


def _cross_validated_knn(values: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.empty(len(target))
    mean_predicted = np.empty(len(target))
    for train, test in KFold(5, shuffle=True, random_state=42).split(values):
        model = KNeighborsRegressor(
            n_neighbors=min(15, len(train)), weights="distance", metric="cosine"
        ).fit(values[train], target[train])
        predicted[test] = model.predict(values[test])
        mean_predicted[test] = target[train].mean()
    return predicted, mean_predicted


def _held_out_behavior(features: pd.DataFrame, champion: dict) -> list[dict]:
    rows = []
    base_names = champion["feature_names"]
    cohort_mask = features["offensive_eligible"].fillna(False) & features["season"].eq("2024-25")
    for target, excluded in HOLDOUT_TARGETS.items():
        names = [name for name in base_names if name not in excluded]
        artifact = fit_style_artifact(
            features, "Offensive", profile="Broad history", training_end="2023-24",
            latent_dimensions=min(16, len(names)), feature_names=names,
            feature_weights={name: .25 for name in BROAD_V2_SHOT_FEATURES},
        )
        cohort = features.loc[cohort_mask & features[target].notna()].reset_index(drop=True)
        actual = cohort[target].to_numpy(dtype=float)
        for method in METHODS:
            values = transform_style(artifact, cohort, method)
            predicted, mean_predicted = _cross_validated_knn(values, actual)
            scale = float(np.std(actual)) or 1.0
            rows.append({
                "target": target, "method": method, "eligible_rows": len(cohort),
                "mae": float(mean_absolute_error(actual, predicted)),
                "normalized_mae": float(mean_absolute_error(actual, predicted) / scale),
                "r2": float(r2_score(actual, predicted)),
                "mean_baseline_mae": float(mean_absolute_error(actual, mean_predicted)),
                "beats_mean_baseline": bool(
                    mean_absolute_error(actual, predicted)
                    < mean_absolute_error(actual, mean_predicted)
                ),
                "excluded_features": sorted(excluded),
            })
        print(f"Held-out behavior: {target}")
    return rows


def _weighted_features(names: list[str]) -> dict[str, float]:
    return {name: .25 for name in names if name in BROAD_V2_SHOT_FEATURES}


def _identity_diagnostics(
    features: pd.DataFrame,
    champion: dict,
    split_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> list[dict]:
    feature_sets = {
        "full_candidate": champion["feature_names"],
        "without_lateral_location": [
            name for name in champion["feature_names"] if name not in LATERAL_FEATURES
        ],
        "without_fine_distance": [
            name for name in champion["feature_names"] if name not in FINE_DISTANCE_FEATURES
        ],
        "without_action_labels": [
            name for name in champion["feature_names"] if name not in ACTION_LABEL_FEATURES
        ],
        "lateral_location_only": sorted(LATERAL_FEATURES),
        "fine_distance_only": sorted(FINE_DISTANCE_FEATURES),
        "action_labels_only": sorted(ACTION_LABEL_FEATURES),
    }
    rows = []
    for variant, names in feature_sets.items():
        if variant == "full_candidate":
            artifact = champion
        else:
            artifact = fit_style_artifact(
                features, "Offensive", profile="Broad history", training_end="2023-24",
                latent_dimensions=min(16, len(names)), feature_names=names,
                feature_weights=_weighted_features(names),
            )
        embeddings = build_embedding_frame(features, artifact, "Denoising autoencoder")
        adjacent = evaluate_temporal_retrieval(
            embeddings, query_season="2025-26", candidate_season="2024-25"
        )
        rows.append({"variant": variant, "evaluation": "adjacent_season", **adjacent})
        for season in SPLIT_SEASONS:
            first, second = split_frames[season]
            rows.append({
                "variant": variant, "evaluation": "split_season", "season": season,
                **_split_metrics(first, second, artifact, "Denoising autoencoder"),
            })
        print(f"Identity diagnostic: {variant}")
    return rows


def _player_checks(embeddings: pd.DataFrame) -> dict[str, list[dict]]:
    checks = {}
    for name in [
        "Shai Gilgeous-Alexander", "Stephen Curry", "Nikola Jokic",
        "Rudy Gobert", "Luka Doncic",
    ]:
        reference = embeddings.loc[
            embeddings["player_name"].eq(name) & embeddings["season"].eq("2025-26")
        ]
        if reference.empty:
            checks[name] = []
            continue
        neighbors = find_style_neighbors(
            embeddings, int(reference.iloc[0]["player_id"]), lens="Offensive",
            method="Denoising autoencoder", profile="Broad history", season="2025-26",
            candidate_season_start="1996-97", candidate_season_end="2025-26", top_n=100,
        ).drop_duplicates("player_id").head(5)
        checks[name] = neighbors[
            ["player_name", "season", "similarity_score", "cosine_similarity"]
        ].to_dict(orient="records")
    return checks


def main() -> int:
    args = _arguments()
    root = settings.historical_data_dir
    champion = joblib.load(args.candidate_dir / "model.joblib")
    features = pd.read_parquet(root / "model_registry" / "play_style" / "v1" / "player_style_features.parquet")
    location = pd.read_parquet(args.candidate_dir / "broad_v2_shot_profiles.parquet")
    features = features.merge(location, on=["player_id", "season"], how="left", validate="one_to_one")

    split_frames, split_rows = {}, []
    for season in SPLIT_SEASONS:
        print(f"Building split-season samples for {season}")
        split_frames[season] = _split_season(root, season)
        for method in METHODS:
            split_rows.append({
                "season": season, "method": method,
                **_split_metrics(*split_frames[season], champion, method),
            })

    bootstrap_summary, bootstrap_players = _bootstrap(
        root, BOOTSTRAP_SEASON, champion, args.bootstrap_repeats
    )
    held_out = _held_out_behavior(features, champion)
    identity = _identity_diagnostics(features, champion, split_frames)
    embeddings = pd.read_parquet(args.candidate_dir / "embeddings.parquet")
    players = _player_checks(embeddings)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": "broad-history-v2-candidate",
        "model_changed": False,
        "split_season": split_rows,
        "bootstrap": bootstrap_summary,
        "held_out_behavior": held_out,
        "identity_fingerprints": identity,
        "representative_player_checks": players,
        "notes": [
            "Split-season halves require 10 games, 250 minutes, and 50 shots per half.",
            "Bootstrap resamples complete player-games with replacement and recomputes every input.",
            "Held-out behavior removes the target and direct aliases before five-fold kNN prediction.",
            "Identity-only feature results quantify shortcut risk; they are not proof that a real tendency is invalid.",
        ],
    }
    args.candidate_dir.mkdir(parents=True, exist_ok=True)
    output = args.candidate_dir / "robustness_evaluation.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    bootstrap_players.to_parquet(
        args.candidate_dir / "bootstrap_player_stability.parquet", index=False, compression="snappy"
    )
    tracked = ROOT / "models" / "play_style" / "broad_v2_robustness_summary.json"
    tracked.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {tracked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
