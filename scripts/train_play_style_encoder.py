"""Train and evaluate offensive, defensive, and overall player-style encoders."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ball_ai.analytics.play_style import (  # noqa: E402
    OFFENSIVE_ACTION_FEATURES,
    STYLE_FEATURE_SETS,
    build_action_profiles,
    build_defensive_matchup_profiles,
    build_embedding_frame,
    build_shot_context_features,
    build_style_features,
    evaluate_temporal_retrieval,
    fit_style_artifact,
)
from ball_ai.config import settings  # noqa: E402


METHODS = ["Cosine baseline", "PCA baseline", "Denoising autoencoder"]
LATENT_CANDIDATES = {
    "Offensive": [12, 16, 20, 24],
    "Defensive": [8, 10, 12, 14],
    "Overall": [20, 24, 28, 32, 36],
}
ACTION_WEIGHT_CANDIDATES = [0.0, 0.25, 0.5, 0.75, 1.0]


def _load_matchups(root: Path) -> pd.DataFrame:
    paths = sorted((root / "matchup_events").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            "Matchup data is missing. Run scripts/sync_play_style_data.py first."
        )
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def _load_actions(root: Path) -> pd.DataFrame:
    paths = sorted((root / "play_by_play_cdn").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            "Action data is missing. Run scripts/sync_play_style_data.py --include-actions first."
        )
    columns = [
        "actionType", "subType", "qualifiers", "personId", "assistPersonId",
        "possession", "orderNumber", "clock", "period", "gameId", "shotResult",
        "_season", "_season_type",
    ]
    return pd.concat(
        (pd.read_parquet(path, columns=columns) for path in paths), ignore_index=True
    )


def main() -> int:
    root = settings.historical_data_dir
    player_seasons = pd.read_parquet(root / "player_season_stats.parquet")
    player_seasons = player_seasons.loc[player_seasons["season"].ge("1996-97")].copy()
    shot_profiles = pd.read_parquet(root / "player_shot_profiles.parquet")
    shot_profiles = shot_profiles.loc[shot_profiles["season"].ge("1996-97")].copy()
    biographies = pd.read_parquet(root / "players.parquet")
    impact_path = root / "darko_dpm.parquet"
    impact = pd.read_parquet(impact_path) if impact_path.exists() else pd.DataFrame()

    print("Building shot-context features...")
    shot_columns = [
        "PLAYER_ID", "SHOT_MADE_FLAG", "SHOT_DISTANCE", "ACTION_TYPE",
        "SHOT_ZONE_BASIC", "TEAM_ID", "PERIOD", "HTM", "_season", "_season_type",
    ]
    shots = pd.read_parquet(
        root / "shot_events.parquet",
        columns=shot_columns,
        filters=[("_season", ">=", 1996)],
    )
    shot_context = build_shot_context_features(shots)
    print("Building defensive matchup profiles...")
    defensive = build_defensive_matchup_profiles(_load_matchups(root), biographies)
    print("Building possession-action profiles...")
    actions = build_action_profiles(_load_actions(root))
    features = build_style_features(
        player_seasons,
        shot_profiles,
        biographies,
        shot_context=shot_context,
        defensive_matchups=defensive,
        action_profiles=actions,
        impact=impact,
    )
    features.to_parquet(root / "player_style_features.parquet", index=False, compression="snappy")

    artifacts = {profile: {} for profile in STYLE_FEATURE_SETS}
    embedding_frames = []
    evaluations = []
    tuning_results = []
    for profile in STYLE_FEATURE_SETS:
        for lens in STYLE_FEATURE_SETS[profile]:
            print(f"Training {profile.lower()} {lens.lower()} encoder...")
            candidates = []
            action_weights = (
                ACTION_WEIGHT_CANDIDATES
                if profile == "Modern detailed" and lens != "Defensive"
                else [1.0]
            )
            for dimensions in LATENT_CANDIDATES[lens]:
                for action_weight in action_weights:
                    tuned = fit_style_artifact(
                        features,
                        lens,
                        profile=profile,
                        latent_dimensions=dimensions,
                        feature_weights={
                            feature: action_weight for feature in OFFENSIVE_ACTION_FEATURES
                        },
                    )
                    tuned_embeddings = build_embedding_frame(
                        features, tuned, "Denoising autoencoder"
                    )
                    validation = evaluate_temporal_retrieval(
                        tuned_embeddings,
                        query_season="2024-25",
                        candidate_season="2023-24",
                    )
                    tuning_results.append({
                        "profile": profile,
                        "lens": lens,
                        "latent_dimensions": dimensions,
                        "action_weight": action_weight if profile == "Modern detailed" else None,
                        **validation,
                    })
                    candidates.append((validation["mean_reciprocal_rank"], tuned))
            artifact = max(candidates, key=lambda item: item[0])[1]
            artifacts[profile][lens] = artifact
            for method in METHODS:
                embeddings = build_embedding_frame(features, artifact, method)
                embedding_frames.append(embeddings)
                for split, query, candidate_season in (
                    ("validation", "2024-25", "2023-24"),
                    ("test", "2025-26", "2024-25"),
                ):
                    try:
                        metrics = evaluate_temporal_retrieval(
                            embeddings,
                            query_season=query,
                            candidate_season=candidate_season,
                        )
                    except ValueError as exc:
                        metrics = {
                            "error": str(exc), "query_season": query,
                            "candidate_season": candidate_season,
                        }
                    evaluations.append({
                        "profile": profile, "lens": lens, "method": method,
                        "split": split, **metrics,
                    })

    all_embeddings = pd.concat(embedding_frames, ignore_index=True).fillna({
        column: 0.0
        for frame in embedding_frames
        for column in frame.columns
        if column.startswith("embedding_")
    })
    all_embeddings.to_parquet(
        root / "player_style_embeddings.parquet", index=False, compression="snappy"
    )
    joblib.dump(artifacts, root / "play_style_models.joblib")

    evaluation = pd.DataFrame(evaluations)
    selected_methods = {
        profile: (
            evaluation.loc[
                evaluation["split"].eq("validation")
                & evaluation["profile"].eq(profile)
            ]
            .sort_values("mean_reciprocal_rank", ascending=False)
            .groupby("lens", as_index=False)
            .first()
            .set_index("lens")["method"]
            .to_dict()
        )
        for profile in STYLE_FEATURE_SETS
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "Denoising autoencoder",
        "training_end": "2022-23",
        "feature_sets": STYLE_FEATURE_SETS,
        "selected_methods": selected_methods,
        "selected_latent_dimensions": {
            profile: {
                lens: int(artifact["latent_dimensions"])
                for lens, artifact in profile_artifacts.items()
            }
            for profile, profile_artifacts in artifacts.items()
        },
        "selected_action_weights": {
            profile: {
                lens: float(artifact["feature_weights"][
                    artifact["feature_names"].index(OFFENSIVE_ACTION_FEATURES[0])
                ]) if profile == "Modern detailed" and lens != "Defensive" else None
                for lens, artifact in profile_artifacts.items()
            }
            for profile, profile_artifacts in artifacts.items()
        },
        "autoencoder_tuning": tuning_results,
        "eligibility": {
            "Broad history": {
                "offensive": "20 games, 500 minutes, and 100 shot attempts",
                "defensive": "20 games, 500 minutes, 300 partial matchup possessions, and 75 matchup FGA",
                "overall": "Must pass the broad offensive and defensive rules",
            },
            "Modern detailed": {
                "offensive": "Broad offensive eligibility plus 100 identified action shots",
                "defensive": "Defensive eligibility plus 100 identified action shots for a consistent modern cohort",
                "overall": "Must pass the modern offensive and defensive rules",
            },
            "home_away_context": "40 shots in each split",
            "quarter_context": "40 first-half and 40 fourth-quarter shots",
            "playoff_context": "100 regular-season and 30 playoff shots",
        },
        "data_rows": int(len(features)),
        "evaluations": evaluation.to_dict(orient="records"),
        "limitations": [
            "The learned representation is descriptive style, not player quality.",
            "DPM is attached for display and is excluded from every style encoder.",
            "Player identity, team, position, height, weight, and opponent height are excluded from every style encoder.",
            "Broad History excludes possession-action fields and begins with shot-detail coverage in 1996-97.",
            "Modern Detailed requires identified action coverage from 2020-21 onward; missing action seasons are ineligible rather than imputed into its comparisons.",
            "Early/late-clock style is reconstructed from explicit possession changes and offensive-rebound resets; the source does not expose a direct shot-clock field.",
            "Defensive style uses partial matchup tracking from 2017-18 onward, does not observe scheme instructions, and remains lower-confidence than offense or overall style.",
        ],
    }
    (root / "play_style_evaluation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    columns = [
        "profile", "lens", "method", "split", "eligible_queries", "recall_at_1",
        "recall_at_3", "recall_at_5", "recall_at_10", "mean_reciprocal_rank", "median_rank",
    ]
    print(evaluation.reindex(columns=columns).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
