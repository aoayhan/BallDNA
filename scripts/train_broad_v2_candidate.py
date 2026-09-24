"""Build and backtest a stable 1996+ Broad History offensive challenger."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import joblib


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ball_ai.analytics.play_style import (  # noqa: E402
    BROAD_OFFENSIVE_FEATURES,
    build_embedding_frame,
    evaluate_temporal_retrieval,
    fit_style_artifact,
)
from ball_ai.analytics.shot_profile import (  # noqa: E402
    BROAD_V2_SHOT_FEATURES,
    build_broad_v2_shot_profiles,
)
from ball_ai.config import settings  # noqa: E402
from evaluate_play_style_rolling import ROLLING_FOLDS  # noqa: E402


STABLE_BASE_FEATURES = [
    "rim_frequency",
    "paint_frequency",
    "midrange_frequency",
    "three_point_frequency",
    "average_shot_distance",
    "field_goal_attempts_per_36",
    "three_point_attempt_rate",
    "free_throw_attempt_rate",
    "assists_per_36",
    "turnovers_per_36",
    "assist_turnover_ratio",
    "offensive_rebounds_per_36",
    "estimated_used_possessions_per_36",
]
V2_FEATURES = [*STABLE_BASE_FEATURES, *BROAD_V2_SHOT_FEATURES]
AUGMENTED_FEATURES = list(dict.fromkeys([*BROAD_OFFENSIVE_FEATURES, *BROAD_V2_SHOT_FEATURES]))
FINAL_TEST_FOLD = "2025-26 -> 2024-25"


def _arguments() -> argparse.Namespace:
    candidate = settings.historical_data_dir / "model_registry" / "play_style" / "candidates" / "broad_v2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=candidate)
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def _location_profiles(root: Path, destination: Path, rebuild: bool) -> pd.DataFrame:
    if destination.exists() and not rebuild:
        return pd.read_parquet(destination)
    raw_path = root / "shot_events.parquet"
    seasons = pd.read_parquet(root / "player_shot_profiles.parquet", columns=["season"])["season"].unique()
    columns = [
        "PLAYER_ID", "SHOT_MADE_FLAG", "SHOT_DISTANCE", "ACTION_TYPE",
        "SHOT_ZONE_BASIC", "SHOT_TYPE", "SHOT_ZONE_AREA", "LOC_X",
    ]
    profiles = []
    for season in sorted(seasons):
        year = int(str(season)[:4])
        print(f"Building stable shot-location features for {season}")
        raw = pd.read_parquet(
            raw_path,
            columns=columns,
            filters=[[('_season', '=', year), ('_season_type', '=', 'rg')]],
        ).rename(columns={
            "PLAYER_ID": "player_id",
            "SHOT_MADE_FLAG": "shot_made",
            "SHOT_DISTANCE": "shot_distance",
            "ACTION_TYPE": "action_type",
            "SHOT_ZONE_BASIC": "shot_zone_basic",
            "SHOT_TYPE": "shot_type",
            "SHOT_ZONE_AREA": "shot_zone_area",
            "LOC_X": "loc_x",
        })
        raw["player_id"] = pd.to_numeric(raw["player_id"], errors="coerce")
        raw = raw.dropna(subset=["player_id"])
        raw["player_id"] = raw["player_id"].astype(int)
        profile = build_broad_v2_shot_profiles(raw)
        profile.insert(1, "season", str(season))
        profiles.append(profile)
    output = pd.concat(profiles, ignore_index=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(destination, index=False, compression="snappy")
    return output


def _era_relative(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output[V2_FEATURES] = output.groupby("season")[V2_FEATURES].rank(pct=True)
    return output


def main() -> int:
    args = _arguments()
    champion = settings.historical_data_dir / "model_registry" / "play_style" / "v1"
    base = pd.read_parquet(champion / "player_style_features.parquet")
    locations = _location_profiles(
        settings.historical_data_dir,
        args.candidate_dir / "broad_v2_shot_profiles.parquet",
        args.rebuild_features,
    )
    features = base.merge(
        locations,
        on=["player_id", "season"],
        how="left",
        validate="one_to_one",
    )
    variants = [
        ("stable_absolute", features, V2_FEATURES, None, [500, 1500]),
        ("stable_era_relative", _era_relative(features), V2_FEATURES, None, [500, 1500]),
        *[
            (
                f"champion_plus_location_{weight:.2f}",
                features,
                AUGMENTED_FEATURES,
                {feature: weight for feature in BROAD_V2_SHOT_FEATURES},
                [500],
            )
            for weight in (0.25, 0.5, 1.0)
        ],
    ]
    rows = []
    final_artifacts = {}
    for variant, frame, feature_names, feature_weights, training_budgets in variants:
        for max_iter in training_budgets:
            for training_end, query_season, candidate_season in ROLLING_FOLDS:
                print(
                    f"Training {variant} / {max_iter} iterations through {training_end}; "
                    f"testing {query_season} -> {candidate_season}"
                )
                artifact = fit_style_artifact(
                    frame,
                    "Offensive",
                    profile="Broad history",
                    training_end=training_end,
                    latent_dimensions=16,
                    feature_names=feature_names,
                    feature_weights=feature_weights,
                    max_iter=max_iter,
                )
                embeddings = build_embedding_frame(frame, artifact, "Denoising autoencoder")
                metrics = evaluate_temporal_retrieval(
                    embeddings,
                    query_season=query_season,
                    candidate_season=candidate_season,
                )
                rows.append({
                    "variant": variant,
                    "max_iter": max_iter,
                    "location_weight": next(iter(feature_weights.values())) if feature_weights else None,
                    "training_end": training_end,
                    "fold": f"{query_season} -> {candidate_season}",
                    "encoder_iterations": int(artifact["encoder"].n_iter_),
                    "converged_before_limit": int(artifact["encoder"].n_iter_) < max_iter,
                    **metrics,
                })
                if training_end == "2023-24":
                    final_artifacts[(variant, max_iter)] = (artifact, embeddings)

    results = pd.DataFrame(rows)
    metrics = [
        "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
        "mean_reciprocal_rank", "median_rank",
    ]
    summary = results.groupby(["variant", "max_iter"], as_index=False).agg(
        **{metric: (metric, "mean") for metric in metrics},
        folds=("fold", "count"),
        converged_folds=("converged_before_limit", "sum"),
        mean_encoder_iterations=("encoder_iterations", "mean"),
    )
    selection = (
        results.loc[results["fold"].ne(FINAL_TEST_FOLD)]
        .groupby(["variant", "max_iter"], as_index=False)[metrics]
        .mean()
        .sort_values("mean_reciprocal_rank", ascending=False)
    )
    winner = selection.iloc[0]
    winner_key = (str(winner["variant"]), int(winner["max_iter"]))
    final_test = results.loc[
        results["fold"].eq(FINAL_TEST_FOLD)
        & results["variant"].eq(winner_key[0])
        & results["max_iter"].eq(winner_key[1])
    ].iloc[0]
    selected_candidate = {
        "variant": winner_key[0],
        "max_iter": winner_key[1],
        "selection_folds": 8,
        "selection_mean_reciprocal_rank": float(winner["mean_reciprocal_rank"]),
        "selection_recall_at_5": float(winner["recall_at_5"]),
        "final_test_fold": FINAL_TEST_FOLD,
        "final_test_mean_reciprocal_rank": float(final_test["mean_reciprocal_rank"]),
        "final_test_recall_at_5": float(final_test["recall_at_5"]),
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate_model_id": "broad-history-v2",
        "champion_model_id": "play-style-v1",
        "feature_policy": (
            "Uses only box-score and shot-location fields consistently available since 1996-97; "
            "era-dependent move labels are excluded."
        ),
        "feature_sets": {
            "stable_only": V2_FEATURES,
            "champion_plus_location": AUGMENTED_FEATURES,
        },
        "selected_candidate": selected_candidate,
        "folds": [
            {"training_end": train, "query_season": query, "candidate_season": candidate}
            for train, query, candidate in ROLLING_FOLDS
        ],
        "summary": summary.to_dict(orient="records"),
        "evaluations": results.to_dict(orient="records"),
    }
    args.candidate_dir.mkdir(parents=True, exist_ok=True)
    winner_artifact, winner_embeddings = final_artifacts[winner_key]
    joblib.dump(winner_artifact, args.candidate_dir / "model.joblib")
    winner_embeddings.to_parquet(
        args.candidate_dir / "embeddings.parquet", index=False, compression="snappy"
    )
    (args.candidate_dir / "evaluation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    tracked = ROOT / "models" / "play_style" / "broad_v2_candidate_summary.json"
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text(
        json.dumps({key: payload[key] for key in payload if key != "evaluations"}, indent=2),
        encoding="utf-8",
    )
    print("\n", summary.to_string(index=False))
    print(f"\nSaved {tracked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
