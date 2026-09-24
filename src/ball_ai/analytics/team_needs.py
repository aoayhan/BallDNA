"""Explainable team benchmarking and player-fit retrieval.

The model describes which measurable team profiles resemble historically elite
regular-season teams. It is not a causal roster-move or win forecast.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ball_ai.config import settings
from ball_ai.data.kaggle_provider import TEAM_ABBREVIATIONS


TEAM_FEATURES = [
    "effective_field_goal_percentage",
    "opponent_effective_field_goal_percentage",
    "turnover_percentage",
    "forced_turnover_percentage",
    "offensive_rebound_percentage",
    "defensive_rebound_percentage",
    "free_throw_rate",
    "three_point_attempt_rate",
    "assist_ratio",
    "pace",
    "stocks_per_100",
    "paint_scoring_share",
    "fast_break_scoring_share",
    "second_chance_scoring_share",
    "bench_scoring_share",
]

TEAM_FEATURE_LABELS = {
    "effective_field_goal_percentage": "Effective FG%",
    "opponent_effective_field_goal_percentage": "Opponent eFG%",
    "turnover_percentage": "Turnover rate",
    "forced_turnover_percentage": "Forced-turnover rate",
    "offensive_rebound_percentage": "Offensive rebound rate",
    "defensive_rebound_percentage": "Defensive rebound rate",
    "free_throw_rate": "Free-throw rate",
    "three_point_attempt_rate": "Three-point attempt rate",
    "assist_ratio": "Assists per field goal",
    "pace": "Estimated pace",
    "stocks_per_100": "Steals + blocks per 100 possessions",
    "paint_scoring_share": "Paint scoring share",
    "fast_break_scoring_share": "Fast-break scoring share",
    "second_chance_scoring_share": "Second-chance scoring share",
    "bench_scoring_share": "Bench scoring share",
}

PERCENT_FEATURES = {
    "effective_field_goal_percentage",
    "opponent_effective_field_goal_percentage",
    "turnover_percentage",
    "forced_turnover_percentage",
    "offensive_rebound_percentage",
    "defensive_rebound_percentage",
    "free_throw_rate",
    "three_point_attempt_rate",
    "paint_scoring_share",
    "fast_break_scoring_share",
    "second_chance_scoring_share",
    "bench_scoring_share",
}

TEAM_TO_PLAYER_TRAIT = {
    "effective_field_goal_percentage": "shooting_efficiency",
    "opponent_effective_field_goal_percentage": "defensive_activity",
    "turnover_percentage": "decision_making",
    "forced_turnover_percentage": "turnover_creation",
    "offensive_rebound_percentage": "offensive_rebounding",
    "defensive_rebound_percentage": "defensive_rebounding",
    "free_throw_rate": "rim_pressure",
    "three_point_attempt_rate": "perimeter_spacing",
    "assist_ratio": "playmaking",
    "pace": "transition_play",
    "stocks_per_100": "defensive_activity",
    "paint_scoring_share": "interior_scoring",
    "fast_break_scoring_share": "transition_play",
    "second_chance_scoring_share": "offensive_rebounding",
    "bench_scoring_share": "scoring_punch",
}

PLAYER_TRAIT_LABELS = {
    "shooting_efficiency": "efficient shot maker",
    "defensive_activity": "active team defender",
    "decision_making": "low-turnover decision maker",
    "turnover_creation": "turnover creator",
    "offensive_rebounding": "offensive rebounder",
    "defensive_rebounding": "defensive rebounder",
    "rim_pressure": "rim-pressure creator",
    "perimeter_spacing": "high-volume spacer",
    "playmaking": "secondary playmaker",
    "transition_play": "transition contributor",
    "interior_scoring": "interior finisher",
    "scoring_punch": "scoring option",
}


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator and pd.notna(denominator) else np.nan


def _game_id_type(game_ids: pd.Series) -> pd.Series:
    ids = game_ids.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    return pd.Series(
        np.select(
            [ids.str.startswith("002"), ids.str.startswith("004"), ids.str.startswith("001")],
            ["regular", "playoffs", "preseason"],
            default="other",
        ),
        index=game_ids.index,
    )


def load_historical_team_games(
    archive_root: Path | None = None,
    *,
    first_season: str = "2014-15",
    last_season: str | None = None,
) -> pd.DataFrame:
    """Load regular-season team rows, repairing missing 2021-22 type labels."""

    root = Path(archive_root or settings.historical_data_dir) / "team_game_stats"
    if not root.exists():
        raise FileNotFoundError(f"Historical team archive is missing: {root}")
    parts: list[pd.DataFrame] = []
    for season_dir in sorted(root.glob("season=*")):
        season = season_dir.name.split("=", 1)[1]
        if season < first_season or (last_season and season > last_season):
            continue
        for type_dir in season_dir.glob("season_type=*"):
            stored_type = type_dir.name.split("=", 1)[1]
            if stored_type not in {"regular", "unknown"}:
                continue
            for parquet_path in type_dir.glob("*.parquet"):
                frame = pd.read_parquet(parquet_path)
                frame["season"] = season
                frame["stored_season_type"] = stored_type
                parts.append(frame)
    if not parts:
        raise FileNotFoundError("No team game Parquet partitions matched the requested seasons.")
    games = pd.concat(parts, ignore_index=True)
    recovered_type = _game_id_type(games["game_id"])
    keep = games["stored_season_type"].eq("regular") | (
        games["stored_season_type"].eq("unknown") & recovered_type.eq("regular")
    )
    return (
        games.loc[keep]
        .sort_values("game_date")
        .drop_duplicates(["season", "game_id", "team_id"], keep="last")
        .reset_index(drop=True)
    )


def build_team_season_features(games: pd.DataFrame) -> pd.DataFrame:
    """Aggregate team games into possession-adjusted season fingerprints."""

    needed = {
        "season", "game_id", "team_id", "opponent_team_id", "team", "game_date",
        "team_score", "opponent_score", "field_goals_attempted", "field_goals_made",
        "three_pointers_attempted", "three_pointers_made", "free_throws_attempted",
        "free_throws_made", "rebounds_offensive", "rebounds_defensive", "turnovers",
        "assists", "steals", "blocks", "points_in_the_paint", "points_fast_break",
        "points_second_chance", "bench_points",
    }
    if missing := needed - set(games.columns):
        raise ValueError(f"Team game rows are missing columns: {sorted(missing)}")

    games = games.loc[
        pd.to_numeric(games["team_id"], errors="coerce").fillna(0).gt(0)
        & games["team"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    base = [
        "field_goals_attempted", "field_goals_made", "three_pointers_attempted",
        "three_pointers_made", "free_throws_attempted", "free_throws_made",
        "rebounds_offensive", "rebounds_defensive", "turnovers", "assists",
        "steals", "blocks", "team_score", "points_in_the_paint",
        "points_fast_break", "points_second_chance", "bench_points",
    ]
    opponents = games[["season", "game_id", "team_id", *base]].rename(
        columns={"team_id": "opponent_join_id", **{name: f"opponent_{name}" for name in base}}
    )
    paired = games.merge(
        opponents,
        left_on=["season", "game_id", "opponent_team_id"],
        right_on=["season", "game_id", "opponent_join_id"],
        how="left",
        validate="many_to_one",
    )

    rows: list[dict[str, Any]] = []
    sum_columns = [*base, *(f"opponent_{name}" for name in base)]
    for (season, team_id, team), group in paired.groupby(
        ["season", "team_id", "team"], dropna=False
    ):
        totals = group[sum_columns].sum(min_count=1)
        games_played = int(group["game_id"].nunique())
        possessions = (
            totals["field_goals_attempted"]
            - totals["rebounds_offensive"]
            + totals["turnovers"]
            + 0.44 * totals["free_throws_attempted"]
        )
        opponent_possessions = (
            totals["opponent_field_goals_attempted"]
            - totals["opponent_rebounds_offensive"]
            + totals["opponent_turnovers"]
            + 0.44 * totals["opponent_free_throws_attempted"]
        )
        row = {
            "season": str(season),
            "team_id": int(team_id),
            "team": str(team),
            "team_abbreviation": TEAM_ABBREVIATIONS.get(int(team_id), "N/A"),
            "games": games_played,
            "wins": int((group["team_score"] > group["opponent_score"]).sum()),
            "win_percentage": float((group["team_score"] > group["opponent_score"]).mean()),
            "offensive_rating": 100 * _safe_divide(totals["team_score"], possessions),
            "defensive_rating": 100
            * _safe_divide(totals["opponent_team_score"], opponent_possessions),
            "effective_field_goal_percentage": _safe_divide(
                totals["field_goals_made"] + 0.5 * totals["three_pointers_made"],
                totals["field_goals_attempted"],
            ),
            "opponent_effective_field_goal_percentage": _safe_divide(
                totals["opponent_field_goals_made"]
                + 0.5 * totals["opponent_three_pointers_made"],
                totals["opponent_field_goals_attempted"],
            ),
            "turnover_percentage": _safe_divide(totals["turnovers"], possessions),
            "forced_turnover_percentage": _safe_divide(
                totals["opponent_turnovers"], opponent_possessions
            ),
            "offensive_rebound_percentage": _safe_divide(
                totals["rebounds_offensive"],
                totals["rebounds_offensive"] + totals["opponent_rebounds_defensive"],
            ),
            "defensive_rebound_percentage": _safe_divide(
                totals["rebounds_defensive"],
                totals["rebounds_defensive"] + totals["opponent_rebounds_offensive"],
            ),
            "free_throw_rate": _safe_divide(
                totals["free_throws_made"], totals["field_goals_attempted"]
            ),
            "three_point_attempt_rate": _safe_divide(
                totals["three_pointers_attempted"], totals["field_goals_attempted"]
            ),
            "assist_ratio": _safe_divide(totals["assists"], totals["field_goals_made"]),
            "pace": _safe_divide(possessions + opponent_possessions, 2 * games_played),
            "stocks_per_100": 100
            * _safe_divide(totals["steals"] + totals["blocks"], possessions),
            "paint_scoring_share": _safe_divide(
                totals["points_in_the_paint"], totals["team_score"]
            ),
            "fast_break_scoring_share": _safe_divide(
                totals["points_fast_break"], totals["team_score"]
            ),
            "second_chance_scoring_share": _safe_divide(
                totals["points_second_chance"], totals["team_score"]
            ),
            "bench_scoring_share": _safe_divide(totals["bench_points"], totals["team_score"]),
        }
        row["net_rating"] = row["offensive_rating"] - row["defensive_rating"]
        rows.append(row)

    seasons = pd.DataFrame(rows)
    seasons["league_rank"] = seasons.groupby("season")["win_percentage"].rank(
        ascending=False, method="first"
    ).astype(int)
    return seasons.sort_values(["season", "league_rank"]).reset_index(drop=True)


def add_season_standard_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Add within-season z scores to make eras comparable."""

    frame = features.copy()
    for feature in TEAM_FEATURES:
        grouped = frame.groupby("season")[feature]
        means = grouped.transform("mean")
        deviations = grouped.transform("std").replace(0, np.nan)
        frame[f"z__{feature}"] = (frame[feature] - means) / deviations
    return frame


def _model_builders() -> dict[str, Callable[[], Pipeline]]:
    return {
        "Explainable logistic regression": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=2_000, class_weight="balanced"),
        ),
        "Random forest": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=400,
                max_depth=4,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
        ),
        "Gradient boosting": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=150, max_depth=3, l2_regularization=2, random_state=42
            ),
        ),
    }


def evaluate_team_models(
    features: pd.DataFrame,
    *,
    top_n: int = 8,
    validation_seasons: tuple[str, ...] = ("2022-23", "2023-24", "2024-25"),
) -> pd.DataFrame:
    """Compare models using rolling, future-season validation."""

    frame = add_season_standard_scores(features.loc[features["games"] >= 60])
    columns = [f"z__{feature}" for feature in TEAM_FEATURES]
    frame["elite"] = frame["league_rank"].le(top_n).astype(int)
    results: list[dict[str, Any]] = []
    for model_name, builder in _model_builders().items():
        season_scores = []
        for test_season in validation_seasons:
            train = frame.loc[frame["season"] < test_season]
            test = frame.loc[frame["season"].eq(test_season)]
            if train.empty or test.empty or test["elite"].nunique() < 2:
                continue
            model = builder()
            model.fit(train[columns], train["elite"])
            probabilities = model.predict_proba(test[columns])[:, 1]
            predicted_top = test.assign(probability=probabilities).nlargest(top_n, "probability")
            season_scores.append(
                {
                    "roc_auc": roc_auc_score(test["elite"], probabilities),
                    "average_precision": average_precision_score(test["elite"], probabilities),
                    "top_n_precision": float(predicted_top["elite"].mean()),
                }
            )
        scores = pd.DataFrame(season_scores)
        results.append(
            {
                "model": model_name,
                "roc_auc": scores["roc_auc"].mean(),
                "average_precision": scores["average_precision"].mean(),
                "top_n_precision": scores["top_n_precision"].mean(),
                "validation_seasons": len(scores),
            }
        )
    return pd.DataFrame(results).sort_values(
        ["top_n_precision", "average_precision", "roc_auc"], ascending=False
    ).reset_index(drop=True)


def fit_explainable_elite_model(
    features: pd.DataFrame, *, top_n: int = 8
) -> tuple[Pipeline, dict[str, float]]:
    """Fit the selected interpretable model and return feature coefficients."""

    frame = add_season_standard_scores(features.loc[features["games"] >= 60])
    columns = [f"z__{feature}" for feature in TEAM_FEATURES]
    target = frame["league_rank"].le(top_n).astype(int)
    model = _model_builders()["Explainable logistic regression"]()
    model.fit(frame[columns], target)
    logistic = model.named_steps["logisticregression"]
    coefficients = dict(zip(TEAM_FEATURES, logistic.coef_[0].astype(float)))
    return model, coefficients


def analyze_team_needs(
    features: pd.DataFrame,
    *,
    season: str,
    team_id: int,
    top_n: int = 8,
    lookback_seasons: int = 5,
) -> dict[str, Any]:
    """Return elite commonalities and model-weighted gaps for one team."""

    available = sorted(value for value in features["season"].unique() if value <= season)
    selected_seasons = available[-lookback_seasons:]
    window = features.loc[features["season"].isin(selected_seasons)].copy()
    standardized = add_season_standard_scores(window)
    standardized["elite"] = standardized["league_rank"].le(top_n)
    current = standardized.loc[
        standardized["season"].eq(season) & standardized["team_id"].eq(int(team_id))
    ]
    if current.empty:
        raise ValueError(f"Unknown team_id {team_id} for {season}.")
    current_row = current.iloc[0]
    selected_elite = standardized.loc[
        standardized["season"].eq(season) & standardized["elite"]
    ]
    historical_elite = standardized.loc[standardized["elite"]]
    _, coefficients = fit_explainable_elite_model(window, top_n=top_n)
    coefficient_scale = sum(abs(value) for value in coefficients.values()) or 1.0

    rows: list[dict[str, Any]] = []
    for feature in TEAM_FEATURES:
        coefficient = coefficients[feature]
        direction = 1.0 if coefficient >= 0 else -1.0
        target_z = float(historical_elite[f"z__{feature}"].median())
        team_z = float(current_row[f"z__{feature}"])
        gap_z = max(0.0, direction * (target_z - team_z))
        importance = abs(coefficient) / coefficient_scale
        elite_values = historical_elite[f"z__{feature}"].dropna()
        stability = float((direction * elite_values > 0).mean()) if len(elite_values) else 0.0
        rows.append(
            {
                "feature": feature,
                "label": TEAM_FEATURE_LABELS[feature],
                "team_value": float(current_row[feature]),
                "elite_median": float(selected_elite[feature].median()),
                "team_percentile": float(
                    standardized.loc[standardized["season"].eq(season), feature]
                    .rank(pct=True, ascending=direction > 0)
                    .loc[current.index[0]]
                ),
                "coefficient": coefficient,
                "importance": importance,
                "gap_z": gap_z,
                "priority": gap_z * importance * stability,
                "elite_stability": stability,
                "desired_direction": "higher" if direction > 0 else "lower",
                "player_trait": TEAM_TO_PLAYER_TRAIT[feature],
                "player_trait_label": PLAYER_TRAIT_LABELS[TEAM_TO_PLAYER_TRAIT[feature]],
            }
        )
    gaps = pd.DataFrame(rows).sort_values("priority", ascending=False).reset_index(drop=True)
    commonalities = (
        pd.DataFrame(rows)
        .assign(commonality=lambda data: data["importance"] * data["elite_stability"])
        .sort_values("commonality", ascending=False)
        .reset_index(drop=True)
    )
    archetype_traits = (
        gaps.loc[gaps["priority"].gt(0)]
        .groupby(["player_trait", "player_trait_label"], as_index=False)["priority"]
        .sum()
        .sort_values("priority", ascending=False)
    )
    return {
        "team": current_row[
            ["team_id", "team", "team_abbreviation", "season", "league_rank", "games", "wins", "win_percentage",
             "offensive_rating", "defensive_rating", "net_rating"]
        ].to_dict(),
        "benchmark_seasons": selected_seasons,
        "top_n": top_n,
        "gaps": gaps,
        "commonalities": commonalities,
        "archetype_traits": archetype_traits.reset_index(drop=True),
        "coefficients": coefficients,
    }


def load_player_candidate_pool(
    season: str,
    archive_root: Path | None = None,
) -> pd.DataFrame:
    """Join player season, shot style, biography, and most recent team."""

    root = Path(archive_root or settings.historical_data_dir)
    seasons = pd.read_parquet(
        root / "player_season_stats.parquet", filters=[("season", "=", season)]
    )
    seasons.loc[
        pd.to_numeric(seasons["three_point_attempts_total"], errors="coerce")
        .fillna(0)
        .eq(0),
        "three_point_percentage",
    ] = np.nan
    shot_path = root / "player_shot_profiles.parquet"
    if shot_path.exists():
        shots = pd.read_parquet(shot_path, filters=[("season", "=", season)])
        seasons = seasons.merge(shots, on=["player_id", "season"], how="left")

    game_root = root / "player_game_stats" / f"season={season}"
    game_parts: list[pd.DataFrame] = []
    for type_name in ("regular", "unknown"):
        for path in (game_root / f"season_type={type_name}").glob("*.parquet"):
            part = pd.read_parquet(
                path, columns=["game_date", "game_id", "player_id", "team"]
            )
            if type_name == "unknown":
                part = part.loc[_game_id_type(part["game_id"]).eq("regular")]
            game_parts.append(part)
    if game_parts:
        games = pd.concat(game_parts, ignore_index=True).sort_values("game_date")
        last_team = games.groupby("player_id", as_index=False).tail(1)[["player_id", "team"]]
        seasons = seasons.merge(last_team, on="player_id", how="left")

    players_path = root / "players.parquet"
    if players_path.exists():
        players = pd.read_parquet(players_path)
        position_columns = [name for name in ("guard", "forward", "center") if name in players]
        if position_columns:
            labels = {"guard": "G", "forward": "F", "center": "C"}
            players["position"] = players.apply(
                lambda row: "-".join(
                    labels[column] for column in position_columns if row.get(column) == 1
                )
                or "N/A",
                axis=1,
            )
            seasons = seasons.merge(players[["player_id", "position"]], on="player_id", how="left")
    return seasons


def _percentile(frame: pd.DataFrame, column: str, *, higher_is_better: bool = True) -> pd.Series:
    values = (
        pd.to_numeric(frame[column], errors="coerce")
        if column in frame
        else pd.Series(np.nan, index=frame.index)
    )
    ranks = values.rank(pct=True, ascending=higher_is_better)
    return ranks.fillna(0.5)


def build_player_traits(players: pd.DataFrame) -> pd.DataFrame:
    """Create transparent candidate traits on a 0-1 league-percentile scale."""

    frame = players.copy()
    frame["assist_turnover_ratio"] = np.divide(
        frame["assists_per_game"],
        frame["turnovers_per_game"],
        out=np.zeros(len(frame), dtype=float),
        where=frame["turnovers_per_game"].to_numpy(dtype=float) > 0,
    )
    frame["stocks_per_game"] = frame["steals_per_game"] + frame["blocks_per_game"]
    frame["interior_frequency"] = frame[[
        column for column in ("rim_frequency", "paint_frequency") if column in frame
    ]].sum(axis=1, min_count=1)
    frame["finishing_frequency"] = frame[[
        column for column in ("dunk_frequency", "layup_frequency", "floater_frequency")
        if column in frame
    ]].sum(axis=1, min_count=1)

    percentile = lambda column, high=True: _percentile(frame, column, higher_is_better=high)
    traits = {
        "shooting_efficiency": (
            0.75 * percentile("effective_field_goal_percentage")
            + 0.25 * percentile("points_per_game")
        ),
        "defensive_activity": (
            0.50 * percentile("stocks_per_game")
            + 0.25 * percentile("steals_per_game")
            + 0.25 * percentile("defensive_rebounds_per_game")
        ),
        "decision_making": (
            0.55 * percentile("assist_turnover_ratio")
            + 0.30 * percentile("assists_per_game")
            + 0.15 * percentile("turnovers_per_game", False)
        ),
        "turnover_creation": percentile("steals_per_game"),
        "offensive_rebounding": percentile("offensive_rebounds_per_game"),
        "defensive_rebounding": percentile("defensive_rebounds_per_game"),
        "rim_pressure": (
            percentile("free_throw_attempts_per_game") + percentile("interior_frequency")
        ) / 2,
        "perimeter_spacing": (
            percentile("three_point_attempts_per_game")
            + percentile("three_point_frequency")
            + percentile("three_point_percentage")
        ) / 3,
        "playmaking": (
            percentile("assists_per_game") + percentile("assist_turnover_ratio")
        ) / 2,
        "transition_play": (
            percentile("steals_per_game") + percentile("interior_frequency")
        ) / 2,
        "interior_scoring": (
            percentile("interior_frequency")
            + percentile("finishing_frequency")
            + percentile("two_point_percentage")
        ) / 3,
        "scoring_punch": (
            percentile("points_per_game") + percentile("effective_field_goal_percentage")
        ) / 2,
    }
    for name, values in traits.items():
        frame[f"trait__{name}"] = values.clip(0, 1)
    return frame


def recommend_players(
    players: pd.DataFrame,
    analysis: dict[str, Any],
    *,
    minimum_games: int = 20,
    minimum_minutes: float = 12,
    rotation_only: bool = False,
    top_k: int = 8,
    max_per_position: int = 3,
) -> pd.DataFrame:
    """Rank statistical examples by their alignment with model-identified gaps."""

    candidates = build_player_traits(players)
    team = analysis["team"]
    candidates = candidates.loc[
        candidates["games_played"].ge(minimum_games)
        & candidates["minutes_per_game"].ge(minimum_minutes)
        & candidates["team"].ne(team["team_abbreviation"])
    ].copy()
    if rotation_only:
        candidates = candidates.loc[
            candidates["minutes_per_game"].le(30) & candidates["points_per_game"].le(20)
        ]

    gaps = analysis["gaps"].loc[analysis["gaps"]["priority"].gt(0)].head(6)
    trait_weights = gaps.groupby("player_trait")["priority"].sum()
    if trait_weights.empty or candidates.empty:
        return pd.DataFrame()
    trait_weights = trait_weights / trait_weights.sum()
    candidates["fit_score"] = 0.0
    for trait, weight in trait_weights.items():
        candidates["fit_score"] += candidates[f"trait__{trait}"] * float(weight)
    candidates["data_reliability"] = np.minimum(1.0, candidates["games_played"] / 50.0)
    candidates["fit_score"] *= 0.75 + 0.25 * candidates["data_reliability"]

    important_traits = list(trait_weights.sort_values(ascending=False).index[:3])
    candidates["fit_reasons"] = candidates.apply(
        lambda row: ", ".join(
            f"{PLAYER_TRAIT_LABELS[trait]} {row[f'trait__{trait}']:.0%}ile"
            for trait in important_traits
        ),
        axis=1,
    )
    columns = [
        "player_id", "player_name", "team", "position", "games_played",
        "minutes_per_game", "points_per_game", "rebounds_per_game", "assists_per_game",
        "effective_field_goal_percentage", "three_point_percentage", "fit_score",
        "data_reliability", "fit_reasons",
        *(f"trait__{trait}" for trait in trait_weights.index),
    ]
    ranked = candidates.sort_values(
        ["fit_score", "data_reliability"], ascending=False
    )
    ranked["position_bucket"] = ranked["position"].map(
        lambda value: "C"
        if "C" in str(value).split("-")
        else "G"
        if "G" in str(value).split("-")
        else "F"
    )
    diversified = ranked.loc[
        ranked.groupby("position_bucket").cumcount().lt(max_per_position)
    ].head(top_k)
    return diversified[columns].reset_index(drop=True)


def recommend_consensus_players(
    players: pd.DataFrame,
    features: pd.DataFrame,
    *,
    season: str,
    team_id: int,
    rotation_only: bool = False,
    top_k: int = 8,
) -> pd.DataFrame:
    """Return candidates that survive a small cohort/lookback sensitivity grid."""

    recommendations = []
    for top_n in (5, 8, 10):
        for lookback in (3, 5, 8):
            analysis = analyze_team_needs(
                features,
                season=season,
                team_id=team_id,
                top_n=top_n,
                lookback_seasons=lookback,
            )
            frame = recommend_players(
                players,
                analysis,
                rotation_only=rotation_only,
                top_k=12,
                max_per_position=4,
            )
            if not frame.empty:
                recommendations.append(frame.assign(scenario=f"{top_n}-{lookback}"))
    if not recommendations:
        return pd.DataFrame()

    combined = pd.concat(recommendations, ignore_index=True)
    consensus = combined.groupby("player_id", as_index=False).agg(
        selection_rate=("scenario", "nunique"),
        fit_score=("fit_score", "mean"),
    )
    consensus["selection_rate"] /= 9
    consensus["robust_fit_score"] = consensus["fit_score"] * (
        0.5 + 0.5 * consensus["selection_rate"]
    )
    details = combined.sort_values("fit_score", ascending=False).drop_duplicates("player_id")
    ranked = details.drop(columns=["fit_score", "scenario"]).merge(
        consensus, on="player_id", validate="one_to_one"
    ).sort_values(["robust_fit_score", "selection_rate"], ascending=False)
    ranked["position_bucket"] = ranked["position"].map(
        lambda value: "C"
        if "C" in str(value).split("-")
        else "G"
        if "G" in str(value).split("-")
        else "F"
    )
    return (
        ranked.loc[ranked.groupby("position_bucket").cumcount().lt(3)]
        .head(top_k)
        .drop(columns=["position_bucket"])
        .reset_index(drop=True)
    )


def load_team_model_evaluation(archive_root: Path | None = None) -> dict:
    """Read the persisted model-selection results when available."""

    path = Path(archive_root or settings.historical_data_dir) / "team_model_evaluation.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
