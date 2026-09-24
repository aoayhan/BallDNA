"""Build recent player/team embeddings and validate roster-quality models."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ball_ai.analytics.roster_construction import (  # noqa: E402
    MODEL_FEATURES,
    PLAYER_TRAITS,
    attach_vectors_to_stints,
    build_player_vectors,
    build_roster_training_dataset,
    build_team_embeddings,
    evaluate_roster_models,
    evaluate_win_calibrators,
    load_player_team_stints,
    select_recent_usable_seasons,
)
from ball_ai.config import settings  # noqa: E402


def main() -> None:
    """Create small, reproducible assets consumed by the Streamlit lab."""

    root = settings.historical_data_dir
    team_path = root / "team_season_features.parquet"
    if not team_path.exists():
        raise FileNotFoundError(
            "Team features are missing. Run scripts/build_team_needs_assets.py first."
        )
    team_features = pd.read_parquet(team_path)
    candidate_seasons = sorted(
        team_features.loc[team_features["games"].ge(60), "season"].unique()
    )[-8:]
    stints = load_player_team_stints(candidate_seasons, root)
    selected_seasons = select_recent_usable_seasons(
        team_features, stints, count=5
    )
    if len(selected_seasons) < 5:
        raise RuntimeError(
            f"Only {len(selected_seasons)} recent seasons have broad roster coverage."
        )

    player_seasons = pd.read_parquet(root / "player_season_stats.parquet")
    player_seasons = player_seasons.loc[
        player_seasons["season"].isin(selected_seasons)
        # Keep every player with recorded NBA minutes in the searchable pool.
        # Sample-size shrinkage and data_reliability already protect the model;
        # silently dropping one-to-four-game players made valid players appear
        # missing from the product.
        & player_seasons["games_played"].gt(0)
        & player_seasons["minutes_per_game"].gt(0)
    ].copy()
    shots = pd.read_parquet(root / "player_shot_profiles.parquet")
    shots = shots.loc[shots["season"].isin(selected_seasons)].copy()
    biographies = pd.read_parquet(root / "players.parquet")

    impact_path = root / "darko_dpm.parquet"
    impact = pd.read_parquet(impact_path) if impact_path.exists() else pd.DataFrame()
    vectors = build_player_vectors(player_seasons, shots, biographies, impact)
    selected_stints = stints.loc[stints["season"].isin(selected_seasons)].copy()
    player_team_vectors = attach_vectors_to_stints(vectors, selected_stints)
    team_embeddings = build_team_embeddings(player_team_vectors)
    dataset = build_roster_training_dataset(
        team_embeddings,
        team_features.loc[team_features["season"].isin(selected_seasons)],
    )
    evaluation = evaluate_roster_models(dataset)
    win_calibration = evaluate_win_calibrators(dataset)
    learned = evaluation.loc[evaluation["model"].ne("Season-mean baseline")]
    selected = learned.iloc[0]
    baseline = evaluation.loc[evaluation["model"].eq("Season-mean baseline")].iloc[0]

    vector_columns = [
        "season", "player_id", "player_name", "position", "games_played",
        "minutes_per_game", "minutes_total", "points_per_game",
        "rebounds_per_game", "assists_per_game", "effective_field_goal_percentage",
        "three_point_percentage", "shot_data_available", "data_reliability",
        "rate_quality_proxy", "role_strength", "impact_context", "impact_rating",
        "offensive_impact", "defensive_impact", "impact_source", "quality_proxy",
        "rotation_value",
        *(f"trait__{trait}" for trait in PLAYER_TRAITS),
    ]
    stint_columns = [
        "season", "team", "team_id", "player_id", "player_name", "team_minutes",
        "games_with_team", "last_game_date", "position", "games_played",
        "minutes_per_game", "points_per_game", "rebounds_per_game", "assists_per_game",
        "effective_field_goal_percentage", "three_point_percentage",
        "shot_data_available", "data_reliability", "rate_quality_proxy",
        "role_strength", "impact_context", "impact_rating", "offensive_impact",
        "defensive_impact", "impact_source", "quality_proxy", "rotation_value",
        *(f"trait__{trait}" for trait in PLAYER_TRAITS),
    ]
    vectors[vector_columns].to_parquet(
        root / "roster_player_vectors.parquet", index=False, compression="snappy"
    )
    player_team_vectors[stint_columns].to_parquet(
        root / "roster_player_team_vectors.parquet", index=False, compression="snappy"
    )
    dataset.to_parquet(
        root / "roster_team_embeddings.parquet", index=False, compression="snappy"
    )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seasons": selected_seasons,
        "season_selection": (
            "Five most recent completed seasons with at least 28 teams in both the "
            "team outcomes and player-to-team minute archive."
        ),
        "excluded_recent_seasons": sorted(set(candidate_seasons) - set(selected_seasons)),
        "selected_model": str(selected["model"]),
        "selected_model_mae": float(selected["mae"]),
        "selected_model_rmse": float(selected["rmse"]),
        "selected_model_rank_correlation": float(selected["rank_correlation"]),
        "minimum_detectable_change": float(selected["mae"]),
        "minimum_detectable_change_note": (
            "Until roster-change deltas are validated directly, changes smaller than "
            "the held-out Net Rating MAE are reported as no detectable change."
        ),
        "error_interval": {
            "type": "80th percentile absolute error from leave-one-season-out validation",
            "net_rating_points": float(selected["absolute_error_p80"]),
        },
        "win_calibration": win_calibration,
        "beats_season_mean_baseline": bool(selected["mae"] < baseline["mae"]),
        "training_rows": int(len(dataset)),
        "player_vector_rows": int(len(vectors)),
        "player_team_stint_rows": int(len(player_team_vectors)),
        "player_pool_policy": (
            "All players with at least one recorded regular-season game and positive "
            "minutes are retained; small samples are shrunk and labeled by reliability."
        ),
        "player_traits": PLAYER_TRAITS,
        "player_quality": (
            "DARKO DPM is the primary player-impact rate and is transformed monotonically to "
            "the model's 0-1 quality scale. The documented box-score composite is retained only "
            "as a fallback when a player-season is absent from the public DARKO snapshot."
        ),
        "rotation_policy": (
            "Historical and reference roles use each player's share of actual team minutes, so "
            "injuries, trades and bench appearances remain represented. The share is displayed as "
            "a 240-minute equivalent; no positive incumbent contribution is deleted."
        ),
        "talent_index": (
            "65% minute-weighted rotation quality + 35% unweighted quality of the three best "
            "active players; a non-negative one-feature model makes merit-based additions monotonic."
        ),
        "model_features": MODEL_FEATURES,
        "models": evaluation.to_dict(orient="records"),
        "limitations": [
            "Inputs use same-season player performance and are descriptive, not preseason forecasts.",
            "A roster edit is a counterfactual association and does not estimate transaction causality.",
            "Player defense is limited to rebounding, steals, blocks, and a small team-context plus-minus signal; lineup and matchup context are absent.",
            "Historical team assignments represent minutes actually contributed; traded-player vectors use full-season rates.",
            "Rotation minutes total 240 per team game and are capped at 48 per player.",
        ],
    }
    (root / "roster_model_evaluation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print(f"Seasons: {', '.join(selected_seasons)}")
    print(f"Player vectors: {len(vectors):,}")
    print(f"Player-team stints: {len(player_team_vectors):,}")
    print(f"Team-seasons: {len(dataset):,}")
    print(evaluation.to_string(index=False))
    print(f"Selected model: {selected['model']}")


if __name__ == "__main__":
    main()
