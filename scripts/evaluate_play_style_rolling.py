"""Backtest the frozen play-style architecture with fold-isolated training."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ball_ai.analytics.play_style import (  # noqa: E402
    OFFENSIVE_ACTION_FEATURES,
    build_embedding_frame,
    evaluate_temporal_retrieval,
    fit_style_artifact,
)
from ball_ai.config import settings  # noqa: E402


METHODS = ["Cosine baseline", "PCA baseline", "Denoising autoencoder"]
RECENT_FOLDS = [
    ("2020-21", "2022-23", "2021-22"),
    ("2021-22", "2023-24", "2022-23"),
    ("2022-23", "2024-25", "2023-24"),
    ("2023-24", "2025-26", "2024-25"),
]
HISTORICAL_OFFENSIVE_FOLDS = [
    ("1996-97", "1998-99", "1997-98"),
    ("2001-02", "2003-04", "2002-03"),
    ("2006-07", "2008-09", "2007-08"),
    ("2011-12", "2013-14", "2012-13"),
    ("2016-17", "2018-19", "2017-18"),
]
ROLLING_FOLDS = [*HISTORICAL_OFFENSIVE_FOLDS, *RECENT_FOLDS]


def _folds_for(profile: str, lens: str) -> list[tuple[str, str, str]]:
    if profile == "Broad history" and lens == "Offensive":
        return ROLLING_FOLDS
    return RECENT_FOLDS


def _arguments() -> argparse.Namespace:
    champion = settings.historical_data_dir / "model_registry" / "play_style" / "v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-dir", type=Path, default=champion)
    parser.add_argument("--output", type=Path, default=champion / "rolling_evaluation.json")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "models" / "play_style" / "v1" / "rolling_summary.json",
    )
    return parser.parse_args()


def _summary(frame: pd.DataFrame) -> list[dict]:
    metrics = [
        "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
        "mean_reciprocal_rank", "median_rank",
    ]
    selected = frame.loc[frame["is_champion_method"] & frame["error"].isna()]
    summary = selected.groupby(["profile", "lens", "method"], as_index=False)[metrics].mean()
    summary["folds"] = selected.groupby(["profile", "lens", "method"])["fold"].count().to_numpy()
    return summary.to_dict(orient="records")


def main() -> int:
    args = _arguments()
    evaluation_path = args.champion_dir / "play_style_evaluation.json"
    features_path = args.champion_dir / "player_style_features.parquet"
    if not evaluation_path.exists() or not features_path.exists():
        raise FileNotFoundError("Frozen champion evaluation or feature data is missing.")

    champion = json.loads(evaluation_path.read_text(encoding="utf-8"))
    features = pd.read_parquet(features_path)
    rows = []
    for profile, lenses in champion["selected_latent_dimensions"].items():
        for lens, dimensions in lenses.items():
            action_weight = champion["selected_action_weights"][profile][lens]
            weights = (
                {feature: float(action_weight) for feature in OFFENSIVE_ACTION_FEATURES}
                if action_weight is not None
                else None
            )
            for training_end, query_season, candidate_season in _folds_for(profile, lens):
                print(
                    f"Training {profile} / {lens} through {training_end}; "
                    f"testing {query_season} -> {candidate_season}"
                )
                try:
                    artifact = fit_style_artifact(
                        features,
                        lens,
                        profile=profile,
                        training_end=training_end,
                        latent_dimensions=int(dimensions),
                        feature_weights=weights,
                    )
                    for method in METHODS:
                        embeddings = build_embedding_frame(features, artifact, method)
                        metrics = evaluate_temporal_retrieval(
                            embeddings,
                            query_season=query_season,
                            candidate_season=candidate_season,
                        )
                        rows.append({
                            "profile": profile,
                            "lens": lens,
                            "method": method,
                            "is_champion_method": method == champion["selected_methods"][profile][lens],
                            "fold": f"{query_season} -> {candidate_season}",
                            "training_end": training_end,
                            "latent_dimensions": int(artifact["latent_dimensions"]),
                            "action_weight": action_weight,
                            "error": None,
                            **metrics,
                        })
                except ValueError as exc:
                    rows.append({
                        "profile": profile,
                        "lens": lens,
                        "method": champion["selected_methods"][profile][lens],
                        "is_champion_method": True,
                        "fold": f"{query_season} -> {candidate_season}",
                        "training_end": training_end,
                        "latent_dimensions": int(dimensions),
                        "action_weight": action_weight,
                        "error": str(exc),
                    })

    frame = pd.DataFrame(rows)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "champion_model_id": "play-style-v1",
        "design": (
            "For each fold, refit the frozen v1 architecture only on seasons ending at least "
            "one year before the candidate season, then retrieve the same player's adjacent season. "
            "The v1 architecture was originally selected on 2024-25 -> 2023-24, so this measures "
            "multi-year robustness rather than fully nested hyperparameter selection."
        ),
        "folds": {
            "broad_offensive": [
                {"training_end": train, "query_season": query, "candidate_season": candidate}
                for train, query, candidate in ROLLING_FOLDS
            ],
            "recent": [
                {"training_end": train, "query_season": query, "candidate_season": candidate}
                for train, query, candidate in RECENT_FOLDS
            ],
        },
        "summary": _summary(frame),
        "evaluations": frame.to_dict(orient="records"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps({
            "created_at": payload["created_at"],
            "champion_model_id": payload["champion_model_id"],
            "design": payload["design"],
            "folds": payload["folds"],
            "summary": payload["summary"],
        }, indent=2),
        encoding="utf-8",
    )

    champion_rows = frame.loc[frame["is_champion_method"] & frame["error"].isna()]
    columns = [
        "profile", "lens", "fold", "eligible_queries", "recall_at_1",
        "recall_at_5", "mean_reciprocal_rank", "median_rank",
    ]
    print(champion_rows[columns].to_string(index=False))
    print(f"\nSaved {args.output}")
    print(f"Saved {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
