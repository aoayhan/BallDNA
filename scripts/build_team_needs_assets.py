"""Materialize team-season features and rolling model-selection evidence."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ball_ai.analytics.team_needs import (  # noqa: E402
    build_team_season_features,
    evaluate_team_models,
    load_historical_team_games,
)
from ball_ai.config import settings  # noqa: E402


def main() -> int:
    archive = settings.historical_data_dir
    games = load_historical_team_games(archive, first_season="2014-15")
    features = build_team_season_features(games)
    feature_path = archive / "team_season_features.parquet"
    features.to_parquet(feature_path, index=False, compression="snappy")

    evaluation = evaluate_team_models(features)
    selected = evaluation.iloc[0]
    payload = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "selection_rule": "Highest rolling future-season top-8 precision, then average precision and ROC-AUC.",
        "validation_seasons": ["2022-23", "2023-24", "2024-25"],
        "selected_model": str(selected["model"]),
        "models": evaluation.to_dict(orient="records"),
        "important_note": (
            "This is a descriptive elite-profile classifier, not a causal estimate "
            "that acquiring a recommended player will create wins."
        ),
    }
    (archive / "team_model_evaluation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"features": len(features), **payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
