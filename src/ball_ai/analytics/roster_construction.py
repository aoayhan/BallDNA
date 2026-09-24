"""Roster embeddings, historical validation, and counterfactual simulation.

The feature set is deliberately compact and interpretable. Player vectors are
season-relative trait percentiles; team vectors pool those traits using rotation
minutes and add a few roster-shape features. The resulting model is descriptive:
it estimates the quality historically associated with a roster profile, not the
causal effect of a transaction.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ball_ai.config import settings
from ball_ai.data.kaggle_provider import TEAM_ABBREVIATIONS


PLAYER_TRAITS = [
    "scoring_volume",
    "scoring_efficiency",
    "perimeter_volume",
    "perimeter_accuracy",
    "rim_pressure",
    "shot_creation",
    "playmaking",
    "ball_security",
    "offensive_rebounding",
    "defensive_rebounding",
    "disruption",
    "rim_protection",
    "interior_scoring",
]

PLAYER_TRAIT_LABELS = {
    "scoring_volume": "Scoring volume",
    "scoring_efficiency": "Scoring efficiency",
    "perimeter_volume": "Three-point volume",
    "perimeter_accuracy": "Three-point accuracy",
    "rim_pressure": "Rim pressure",
    "shot_creation": "Self-created offense",
    "playmaking": "Playmaking",
    "ball_security": "Ball security",
    "offensive_rebounding": "Offensive rebounding",
    "defensive_rebounding": "Defensive rebounding",
    "disruption": "Defensive disruption",
    "rim_protection": "Rim protection",
    "interior_scoring": "Interior scoring",
}

PROFILE_CATEGORIES = {
    "Perimeter firepower": ["perimeter_volume", "perimeter_accuracy", "scoring_efficiency"],
    "Self-created offense": ["scoring_volume", "shot_creation"],
    "Rim pressure": ["rim_pressure", "interior_scoring"],
    "Ball movement": ["playmaking", "ball_security"],
    "Possession control": ["ball_security", "offensive_rebounding", "defensive_rebounding"],
    "Interior defensive tools": ["rim_protection", "defensive_rebounding"],
    "Defensive disruption": ["disruption", "rim_protection"],
    "Frontcourt physicality": ["interior_scoring", "offensive_rebounding", "defensive_rebounding"],
}

ROSTER_SHAPE_FEATURES = [
    "rotation_quality",
    "top_three_quality",
    "effective_rotation_size",
    "top_three_minute_share",
    "guard_minute_share",
    "center_minute_share",
    "position_entropy",
    "two_way_balance",
]

MODEL_FEATURES = [*(f"roster__{trait}" for trait in PLAYER_TRAITS), *ROSTER_SHAPE_FEATURES]
TALENT_CEILING_FEATURES = ["talent_index"]
TALENT_ROTATION_WEIGHT = 0.65
TALENT_TOP_THREE_WEIGHT = 0.35


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numer = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    denom = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    return pd.Series(
        np.divide(
            numer,
            denom,
            out=np.full(len(numer), np.nan),
            where=np.isfinite(denom) & (denom != 0),
        ),
        index=numerator.index,
    )


def _season_percentile(
    frame: pd.DataFrame, column: str, *, higher_is_better: bool = True
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    filled = values.groupby(frame["season"]).transform(lambda x: x.fillna(x.median()))
    return filled.groupby(frame["season"]).rank(
        pct=True, ascending=higher_is_better, method="average"
    ).fillna(0.5)


def _position_label(row: pd.Series) -> str:
    labels = [
        label
        for column, label in (("guard", "G"), ("forward", "F"), ("center", "C"))
        if int(row.get(column, 0) or 0) == 1
    ]
    return "-".join(labels) or "N/A"


def build_player_vectors(
    player_seasons: pd.DataFrame,
    shot_profiles: pd.DataFrame | None = None,
    biographies: pd.DataFrame | None = None,
    impact_ratings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create interpretable 0-1 player trait vectors within each season.

    Shooting percentages are empirically shrunk toward the season mean so that
    low-attempt players cannot dominate the accuracy dimensions by chance.
    """

    required = {
        "season", "player_id", "player_name", "games_played", "minutes_per_game",
        "points_total", "field_goal_attempts_total", "field_goals_made_total",
        "three_points_made_total", "three_point_attempts_total",
        "free_throw_attempts_total", "assists_total", "turnovers_total",
        "offensive_rebounds_total", "defensive_rebounds_total", "steals_total",
        "blocks_total", "two_point_percentage", "effective_field_goal_percentage",
        "three_point_percentage",
    }
    if missing := required - set(player_seasons.columns):
        raise ValueError(f"Player seasons are missing columns: {sorted(missing)}")

    frame = player_seasons.copy()
    frame["minutes_total"] = frame["games_played"] * frame["minutes_per_game"]
    minutes = frame["minutes_total"].replace(0, np.nan)
    per_36_columns = {
        "points_per_36": "points_total",
        "field_goal_attempts_per_36": "field_goal_attempts_total",
        "three_point_attempts_per_36": "three_point_attempts_total",
        "free_throw_attempts_per_36": "free_throw_attempts_total",
        "assists_per_36": "assists_total",
        "turnovers_per_36": "turnovers_total",
        "offensive_rebounds_per_36": "offensive_rebounds_total",
        "defensive_rebounds_per_36": "defensive_rebounds_total",
        "steals_per_36": "steals_total",
        "blocks_per_36": "blocks_total",
    }
    for target, source in per_36_columns.items():
        frame[target] = 36 * _safe_divide(frame[source], minutes)

    frame["assist_turnover_ratio"] = _safe_divide(
        frame["assists_total"], frame["turnovers_total"].replace(0, np.nan)
    ).fillna(frame["assists_total"].clip(upper=10))

    if shot_profiles is not None and not shot_profiles.empty:
        shot_columns = [
            "player_id", "season", "shot_attempts", "rim_frequency", "paint_frequency",
            "midrange_frequency", "three_point_frequency", "dunk_frequency",
            "layup_frequency", "floater_frequency", "pull_up_frequency",
            "step_back_frequency", "shot_making_above_expected",
        ]
        available = [column for column in shot_columns if column in shot_profiles]
        frame = frame.merge(
            shot_profiles[available].drop_duplicates(["player_id", "season"]),
            on=["player_id", "season"],
            how="left",
            validate="one_to_one",
        )
    frame["shot_data_available"] = frame.get(
        "shot_attempts", pd.Series(np.nan, index=frame.index)
    ).notna()
    for column in (
        "rim_frequency", "paint_frequency", "midrange_frequency",
        "three_point_frequency", "dunk_frequency", "layup_frequency",
        "floater_frequency", "pull_up_frequency", "step_back_frequency",
        "shot_making_above_expected",
    ):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = frame[column].groupby(frame["season"]).transform(
            lambda values: values.fillna(values.dropna().median())
            if values.notna().any()
            else values.fillna(0.0)
        ).fillna(0.0)

    season_efg = frame.groupby("season")["effective_field_goal_percentage"].transform("mean")
    efg_attempts = pd.to_numeric(frame["field_goal_attempts_total"], errors="coerce").fillna(0)
    frame["shrunk_efg"] = (
        frame["effective_field_goal_percentage"].fillna(season_efg) * efg_attempts
        + season_efg * 150
    ) / (efg_attempts + 150)
    season_three = frame.groupby("season")["three_point_percentage"].transform("mean")
    three_attempts = pd.to_numeric(frame["three_point_attempts_total"], errors="coerce").fillna(0)
    frame["shrunk_three_point_percentage"] = (
        frame["three_point_percentage"].fillna(season_three) * three_attempts
        + season_three * 100
    ) / (three_attempts + 100)
    frame["creation_mix"] = (
        frame["pull_up_frequency"]
        + frame["step_back_frequency"]
        + 0.5 * frame["midrange_frequency"]
    )
    frame["interior_mix"] = (
        frame["rim_frequency"]
        + frame["paint_frequency"]
        + 0.5 * (frame["dunk_frequency"] + frame["layup_frequency"])
    )

    pct = lambda column, high=True: _season_percentile(
        frame, column, higher_is_better=high
    )
    traits = {
        "scoring_volume": pct("points_per_36"),
        "scoring_efficiency": (
            0.8 * pct("shrunk_efg") + 0.2 * pct("shot_making_above_expected")
        ),
        "perimeter_volume": (
            0.7 * pct("three_point_attempts_per_36") + 0.3 * pct("three_point_frequency")
        ),
        "perimeter_accuracy": pct("shrunk_three_point_percentage"),
        "rim_pressure": (
            0.55 * pct("free_throw_attempts_per_36")
            + 0.45 * pct("rim_frequency")
        ),
        "shot_creation": (
            0.55 * pct("field_goal_attempts_per_36") + 0.45 * pct("creation_mix")
        ),
        "playmaking": (
            0.65 * pct("assists_per_36") + 0.35 * pct("assist_turnover_ratio")
        ),
        "ball_security": (
            0.55 * pct("assist_turnover_ratio") + 0.45 * pct("turnovers_per_36", False)
        ),
        "offensive_rebounding": pct("offensive_rebounds_per_36"),
        "defensive_rebounding": pct("defensive_rebounds_per_36"),
        "disruption": pct("steals_per_36"),
        "rim_protection": pct("blocks_per_36"),
        "interior_scoring": (
            0.55 * pct("interior_mix") + 0.45 * pct("two_point_percentage")
        ),
    }
    for name, values in traits.items():
        frame[f"trait__{name}"] = values.clip(0, 1)

    frame["rate_quality_proxy"] = (
        0.16 * frame["trait__scoring_volume"]
        + 0.17 * frame["trait__scoring_efficiency"]
        + 0.13 * frame["trait__playmaking"]
        + 0.10 * frame["trait__ball_security"]
        + 0.10 * frame["trait__perimeter_accuracy"]
        + 0.09 * frame["trait__defensive_rebounding"]
        + 0.12 * frame["trait__disruption"]
        + 0.13 * frame["trait__rim_protection"]
    )
    # Per-36 traits describe what a player does. Small, separately exposed role
    # and plus-minus context signals prevent tiny garbage-time production and a
    # large role on a weak team from being interpreted as portable impact.
    frame["role_strength"] = _season_percentile(frame, "minutes_per_game")
    if "plus_minus_per_game" in frame:
        raw_impact_context = _season_percentile(frame, "plus_minus_per_game")
        impact_reliability = frame["minutes_total"] / (frame["minutes_total"] + 1_000)
        frame["impact_context"] = (
            0.5 + (raw_impact_context - 0.5) * impact_reliability
        ).clip(0, 1)
    else:
        frame["impact_context"] = 0.5
    frame["quality_proxy"] = (
        0.80 * frame["rate_quality_proxy"]
        + 0.10 * frame["role_strength"]
        + 0.10 * frame["impact_context"]
    )
    frame["impact_rating"] = np.nan
    frame["offensive_impact"] = np.nan
    frame["defensive_impact"] = np.nan
    frame["impact_source"] = "box-score fallback"
    if impact_ratings is not None and not impact_ratings.empty:
        required_impact = {"season", "player_id", "dpm", "o_dpm", "d_dpm"}
        if missing := required_impact - set(impact_ratings.columns):
            raise ValueError(f"Impact ratings are missing columns: {sorted(missing)}")
        impact = impact_ratings[
            ["season", "player_id", "dpm", "o_dpm", "d_dpm"]
        ].drop_duplicates(["season", "player_id"], keep="last")
        frame = frame.merge(
            impact,
            on=["season", "player_id"],
            how="left",
            validate="one_to_one",
        )
        available = pd.to_numeric(frame["dpm"], errors="coerce").notna()
        frame.loc[available, "impact_rating"] = frame.loc[available, "dpm"]
        frame.loc[available, "offensive_impact"] = frame.loc[available, "o_dpm"]
        frame.loc[available, "defensive_impact"] = frame.loc[available, "d_dpm"]
        frame.loc[available, "impact_source"] = "DARKO DPM"
        # DPM is a rate impact estimate centered on league average (zero). A
        # smooth monotonic transform preserves its ordering and magnitude while
        # keeping the existing 0-1 roster-feature contract.
        frame.loc[available, "quality_proxy"] = 1 / (
            1 + np.exp(-frame.loc[available, "impact_rating"] / 2)
        )
        frame = frame.drop(columns=["dpm", "o_dpm", "d_dpm"])
    # Allocation and outcome scoring must use the same ordering; otherwise a
    # newcomer can win minutes under one proxy and lower the other proxy.
    frame["rotation_value"] = frame["quality_proxy"]
    shot_attempt_values = (
        pd.to_numeric(frame["shot_attempts"], errors="coerce").fillna(0)
        if "shot_attempts" in frame
        else pd.Series(0.0, index=frame.index)
    )
    frame["data_reliability"] = (
        0.45 * np.minimum(1.0, frame["minutes_total"] / 1_200)
        + 0.35 * np.minimum(1.0, frame["games_played"] / 55)
        + 0.20
        * np.minimum(
            1.0,
            shot_attempt_values / 300,
        )
    )

    if biographies is not None and not biographies.empty:
        bio = biographies.copy()
        bio["position"] = bio.apply(_position_label, axis=1)
        columns = ["player_id", "position"]
        for name in ("heightInches", "bodyWeightLbs"):
            if name in bio:
                columns.append(name)
        frame = frame.merge(
            bio[columns].drop_duplicates("player_id"),
            on="player_id",
            how="left",
            validate="many_to_one",
        )
    if "position" not in frame:
        frame["position"] = "N/A"
    frame["position"] = frame["position"].fillna("N/A")
    return frame


def load_player_team_stints(
    seasons: Iterable[str], archive_root: Path | None = None
) -> pd.DataFrame:
    """Aggregate recent regular-season player minutes by team stint."""

    root = Path(archive_root or settings.historical_data_dir) / "player_game_stats"
    parts: list[pd.DataFrame] = []
    for season in seasons:
        directory = root / f"season={season}" / "season_type=regular"
        season_parts = [
            pd.read_parquet(
                path,
                columns=["game_date", "game_id", "player_id", "player_name", "team", "minutes"],
            )
            for path in directory.glob("*.parquet")
        ]
        if season_parts:
            frame = pd.concat(season_parts, ignore_index=True)
            frame["season"] = season
            parts.append(frame)
    if not parts:
        return pd.DataFrame()
    games = pd.concat(parts, ignore_index=True)
    games = games.loc[
        games["team"].isin(set(TEAM_ABBREVIATIONS.values()))
        & pd.to_numeric(games["minutes"], errors="coerce").fillna(0).gt(0)
    ].copy()
    games["minutes"] = pd.to_numeric(games["minutes"], errors="coerce").fillna(0)
    stints = (
        games.groupby(["season", "team", "player_id", "player_name"], as_index=False)
        .agg(
            team_minutes=("minutes", "sum"),
            games_with_team=("game_id", "nunique"),
            last_game_date=("game_date", "max"),
        )
    )
    inverse = {abbreviation: team_id for team_id, abbreviation in TEAM_ABBREVIATIONS.items()}
    stints["team_id"] = stints["team"].map(inverse).astype("int64")
    return stints


def attach_vectors_to_stints(
    player_vectors: pd.DataFrame, stints: pd.DataFrame
) -> pd.DataFrame:
    """Join one season-level player vector to every team stint."""

    return stints.merge(
        player_vectors,
        on=["season", "player_id", "player_name"],
        how="inner",
        validate="many_to_one",
    )


def _position_shares(group: pd.DataFrame, weights: np.ndarray) -> tuple[float, float, float]:
    positions = group["position"].fillna("N/A").astype(str)
    guard = float(weights[positions.str.contains("G").to_numpy()].sum())
    center = float(weights[positions.str.contains("C").to_numpy()].sum())
    forward = max(0.0, 1.0 - guard - center)
    raw = np.array([guard, forward, center], dtype=float)
    raw = raw / raw.sum() if raw.sum() else np.array([1 / 3, 1 / 3, 1 / 3])
    entropy = float(-(raw * np.log(raw + 1e-12)).sum() / np.log(3))
    return guard, center, entropy


def roster_embedding(
    roster: pd.DataFrame,
    *,
    minute_column: str = "team_minutes",
) -> dict[str, float]:
    """Pool player vectors into one interpretable team embedding."""

    if roster.empty:
        raise ValueError("A roster needs at least one player.")
    required = {minute_column, "position", "quality_proxy", *(f"trait__{x}" for x in PLAYER_TRAITS)}
    if missing := required - set(roster.columns):
        raise ValueError(f"Roster rows are missing columns: {sorted(missing)}")
    minutes = pd.to_numeric(roster[minute_column], errors="coerce").fillna(0).clip(lower=0)
    if minutes.sum() <= 0:
        raise ValueError("Roster minutes must sum to more than zero.")
    weights = (minutes / minutes.sum()).to_numpy(dtype=float)
    row: dict[str, float] = {}
    for trait in PLAYER_TRAITS:
        values = pd.to_numeric(roster[f"trait__{trait}"], errors="coerce").fillna(0.5)
        row[f"roster__{trait}"] = float(np.average(values, weights=weights))

    quality = pd.to_numeric(roster["quality_proxy"], errors="coerce").fillna(0.5).to_numpy()
    row["rotation_quality"] = float(np.average(quality, weights=weights))
    active_indices = np.flatnonzero(weights > 0)
    active_quality = quality[active_indices]
    top_indices = active_indices[
        np.argsort(active_quality)[-min(3, len(active_indices)) :]
    ]
    row["top_three_quality"] = float(quality[top_indices].mean())
    row["effective_rotation_size"] = float(1 / np.sum(np.square(weights)))
    row["top_three_minute_share"] = float(np.sort(weights)[-min(3, len(weights)) :].sum())
    guard, center, entropy = _position_shares(roster, weights)
    row["guard_minute_share"] = guard
    row["center_minute_share"] = center
    row["position_entropy"] = entropy
    offense = np.mean(
        [
            row["roster__scoring_volume"], row["roster__scoring_efficiency"],
            row["roster__shot_creation"], row["roster__playmaking"],
        ]
    )
    defense = np.mean(
        [
            row["roster__defensive_rebounding"], row["roster__disruption"],
            row["roster__rim_protection"],
        ]
    )
    row["two_way_balance"] = float(min(offense, defense))
    return row


def build_team_embeddings(player_team_vectors: pd.DataFrame) -> pd.DataFrame:
    """Build one roster embedding per team-season from contributed minutes."""

    rows: list[dict[str, Any]] = []
    for (season, team_id, team), group in player_team_vectors.groupby(
        ["season", "team_id", "team"], sort=True
    ):
        row: dict[str, Any] = {
            "season": str(season),
            "team_id": int(team_id),
            "team_abbreviation": str(team),
            "players_used": int(group["player_id"].nunique()),
            "roster_minutes": float(group["team_minutes"].sum()),
        }
        # Historical teams already contain the actual player-minutes contributed.
        # Reallocating those minutes invents a rotation and can erase valid bench
        # players; their contribution shares naturally sum to one here.
        row.update(roster_embedding(group, minute_column="team_minutes"))
        rows.append(row)
    return pd.DataFrame(rows)


def select_recent_usable_seasons(
    team_features: pd.DataFrame,
    stints: pd.DataFrame,
    *,
    count: int = 5,
    minimum_teams: int = 28,
) -> list[str]:
    """Select the most recent seasons with broad player-to-team coverage."""

    complete = set(
        team_features.loc[team_features["games"].ge(60)]
        .groupby("season")["team_id"]
        .nunique()
        .loc[lambda values: values.ge(minimum_teams)]
        .index.astype(str)
    )
    assigned = set(
        stints.groupby("season")["team_id"]
        .nunique()
        .loc[lambda values: values.ge(minimum_teams)]
        .index.astype(str)
    )
    return sorted(complete & assigned)[-count:]


def build_roster_training_dataset(
    team_embeddings: pd.DataFrame, team_features: pd.DataFrame
) -> pd.DataFrame:
    """Join roster inputs to observed team outcomes without using outcome columns as inputs."""

    targets = team_features[
        [
            "season", "team_id", "team", "team_abbreviation", "games", "wins",
            "win_percentage", "offensive_rating", "defensive_rating", "net_rating",
            "league_rank",
        ]
    ].copy()
    dataset = team_embeddings.merge(
        targets,
        on=["season", "team_id", "team_abbreviation"],
        how="inner",
        validate="one_to_one",
    )
    return dataset.loc[dataset["games"].ge(60)].sort_values(
        ["season", "league_rank"]
    ).reset_index(drop=True)


def _model_builders() -> dict[str, Callable[[], Pipeline]]:
    return {
        "Season-mean baseline": lambda: make_pipeline(
            SimpleImputer(strategy="median"), DummyRegressor(strategy="mean")
        ),
        "Ridge regression": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=12.0)
        ),
        "Elastic net": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            ElasticNet(alpha=0.08, l1_ratio=0.15, max_iter=10_000, random_state=42),
        ),
        "Random forest": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=500,
                max_depth=4,
                min_samples_leaf=5,
                max_features=0.75,
                random_state=42,
                n_jobs=None,
            ),
        ),
        "Gradient boosting": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=180,
                max_depth=3,
                learning_rate=0.05,
                l2_regularization=3.0,
                random_state=42,
            ),
        ),
    }


def evaluate_roster_models(dataset: pd.DataFrame) -> pd.DataFrame:
    """Compare regressors with leave-one-season-out validation."""

    if len(dataset["season"].unique()) < 3:
        raise ValueError("At least three seasons are required for model validation.")
    rows: list[dict[str, Any]] = []
    for name, builder in _model_builders().items():
        actual_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        for test_season in sorted(dataset["season"].unique()):
            train = dataset.loc[dataset["season"].ne(test_season)]
            test = dataset.loc[dataset["season"].eq(test_season)]
            model = builder()
            model.fit(train[MODEL_FEATURES], train["net_rating"])
            prediction_parts.append(model.predict(test[MODEL_FEATURES]))
            actual_parts.append(test["net_rating"].to_numpy(dtype=float))
        actual = np.concatenate(actual_parts)
        predicted = np.concatenate(prediction_parts)
        residuals = actual - predicted
        rank_correlation = float(
            pd.Series(actual).rank().corr(pd.Series(predicted).rank())
        )
        rows.append(
            {
                "model": name,
                "mae": float(mean_absolute_error(actual, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                "rank_correlation": rank_correlation,
                "residual_std": float(np.std(residuals, ddof=1)),
                "absolute_error_p80": float(np.quantile(np.abs(residuals), 0.8)),
                "validation_rows": int(len(actual)),
                "validation_seasons": int(dataset["season"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mae", "rmse"], ascending=True
    ).reset_index(drop=True)


def fit_roster_quality_model(dataset: pd.DataFrame, model_name: str) -> Pipeline:
    """Fit one validated roster-quality model on all available team-seasons."""

    builders = _model_builders()
    if model_name not in builders:
        raise ValueError(f"Unknown roster model: {model_name}")
    model = builders[model_name]()
    model.fit(dataset[MODEL_FEATURES], dataset["net_rating"])
    return model


class BoundedWinCalibrator:
    """Convert Net Rating to bounded expected wins through win probability."""

    def __init__(self, model: LogisticRegression) -> None:
        self.model = model

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return 82-game expected wins for each supplied Net Rating."""

        return 82 * self.model.predict_proba(frame[["net_rating"]])[:, 1]


def fit_win_calibrator(dataset: pd.DataFrame) -> BoundedWinCalibrator:
    """Fit a bounded logistic mapping from observed Net Rating to 82-game wins."""

    ratings: list[float] = []
    outcomes: list[int] = []
    for row in dataset.itertuples():
        games = int(row.games)
        wins = int(row.wins)
        ratings.extend([float(row.net_rating)] * games)
        outcomes.extend([1] * wins + [0] * (games - wins))
    model = LogisticRegression(C=1_000_000, random_state=42)
    model.fit(pd.DataFrame({"net_rating": ratings}), outcomes)
    return BoundedWinCalibrator(model)


def evaluate_win_calibrators(dataset: pd.DataFrame) -> dict[str, float]:
    """Compare bounded and linear Net Rating-to-win mappings by held-out season."""

    actual_parts: list[np.ndarray] = []
    bounded_parts: list[np.ndarray] = []
    linear_parts: list[np.ndarray] = []
    for season in sorted(dataset["season"].unique()):
        train = dataset.loc[dataset["season"].ne(season)]
        test = dataset.loc[dataset["season"].eq(season)]
        actual = 82 * test["wins"] / test["games"]
        bounded = fit_win_calibrator(train)
        linear = LinearRegression().fit(
            train[["net_rating"]], 82 * train["wins"] / train["games"]
        )
        actual_parts.append(actual.to_numpy(dtype=float))
        bounded_parts.append(bounded.predict(test[["net_rating"]]))
        linear_parts.append(linear.predict(test[["net_rating"]]))
    actual_values = np.concatenate(actual_parts)
    return {
        "bounded_logistic_mae": float(
            mean_absolute_error(actual_values, np.concatenate(bounded_parts))
        ),
        "linear_mae": float(
            mean_absolute_error(actual_values, np.concatenate(linear_parts))
        ),
    }


def fit_talent_ceiling_model(dataset: pd.DataFrame) -> Pipeline:
    """Fit an intentionally optimistic, monotonic talent-only association.

    This lens is designed for roster sandbox exploration, not as a replacement
    for the lower-error evidence model. Positive coefficients guarantee that
    improving rotation or top-three talent cannot lower the displayed ceiling.
    """

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearRegression(positive=True),
    )
    model.fit(
        pd.DataFrame(
            {
                "talent_index": (
                    TALENT_ROTATION_WEIGHT * dataset["rotation_quality"]
                    + TALENT_TOP_THREE_WEIGHT * dataset["top_three_quality"]
                )
            }
        ),
        dataset["net_rating"],
    )
    return model


def predict_talent_ceiling(
    embedding: dict[str, float] | pd.Series,
    model: Pipeline,
    win_calibrator: Any,
) -> dict[str, float]:
    """Map the optimistic talent lens to a bounded 82-game sandbox record."""

    frame = pd.DataFrame(
        {
            "talent_index": [
                TALENT_ROTATION_WEIGHT * float(embedding["rotation_quality"])
                + TALENT_TOP_THREE_WEIGHT * float(embedding["top_three_quality"])
            ]
        }
    )
    net_rating = float(model.predict(frame)[0])
    wins = float(
        np.clip(
            win_calibrator.predict(pd.DataFrame({"net_rating": [net_rating]}))[0],
            0,
            82,
        )
    )
    return {
        "net_rating": net_rating,
        "expected_wins": wins,
        "display_wins": int(np.clip(np.rint(wins), 0, 82)),
    }


def predict_roster_quality(
    embedding: dict[str, float] | pd.Series,
    model: Pipeline,
    win_calibrator: Any,
    *,
    net_rating_error: float,
) -> dict[str, float]:
    """Return quality estimates and an empirical out-of-season error interval."""

    frame = pd.DataFrame([{feature: float(embedding[feature]) for feature in MODEL_FEATURES}])
    net_rating = float(model.predict(frame)[0])
    lower_net = net_rating - float(net_rating_error)
    upper_net = net_rating + float(net_rating_error)
    wins = float(win_calibrator.predict(pd.DataFrame({"net_rating": [net_rating]}))[0])
    win_bounds = win_calibrator.predict(
        pd.DataFrame({"net_rating": [lower_net, upper_net]})
    )
    return {
        "net_rating": net_rating,
        "net_rating_low": lower_net,
        "net_rating_high": upper_net,
        "expected_wins": float(np.clip(wins, 0, 82)),
        "wins_low": float(np.clip(min(win_bounds), 0, 82)),
        "wins_high": float(np.clip(max(win_bounds), 0, 82)),
    }


def assess_embedding_support(
    embedding: dict[str, float] | pd.Series,
    dataset: pd.DataFrame,
    *,
    quantile: float = 0.95,
) -> dict[str, Any]:
    """Check whether a simulated roster resembles the model's training data.

    The support boundary is learned without an arbitrary basketball threshold:
    each historical team is compared with its nearest *other* team in standardized
    feature space, and the requested quantile of those distances becomes the
    acceptance limit. A counterfactual beyond that limit is extrapolation, so its
    regression output should not be presented as a supported estimate.
    """

    if not 0 < quantile < 1:
        raise ValueError("Support quantile must be between zero and one.")
    if len(dataset) < 3:
        raise ValueError("At least three historical teams are required.")

    matrix = dataset[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(imputer.fit_transform(matrix))
    query = pd.DataFrame(
        [{feature: float(embedding[feature]) for feature in MODEL_FEATURES}]
    )
    query_standardized = scaler.transform(imputer.transform(query))[0]
    query_distances = np.sqrt(
        np.square(standardized - query_standardized).sum(axis=1)
    )

    pairwise = standardized[:, None, :] - standardized[None, :, :]
    pairwise_distances = np.sqrt(np.square(pairwise).sum(axis=2))
    np.fill_diagonal(pairwise_distances, np.inf)
    historical_nearest = pairwise_distances.min(axis=1)
    threshold = float(np.quantile(historical_nearest, quantile))
    nearest_distance = float(query_distances.min())
    supported = bool(nearest_distance <= threshold)

    violations = []
    for feature in MODEL_FEATURES:
        values = matrix[feature].dropna()
        value = float(embedding[feature])
        minimum = float(values.min())
        maximum = float(values.max())
        if value < minimum or value > maximum:
            violations.append(
                {
                    "feature": feature,
                    "value": value,
                    "training_min": minimum,
                    "training_max": maximum,
                    "direction": "below" if value < minimum else "above",
                }
            )

    nearest_index = int(np.argmin(query_distances))
    nearest = dataset.iloc[nearest_index]
    return {
        "status": "supported" if supported else "out_of_distribution",
        "is_supported": supported,
        "nearest_distance": nearest_distance,
        "support_threshold": threshold,
        "distance_ratio": nearest_distance / threshold if threshold else float("inf"),
        "threshold_quantile": quantile,
        "features_outside_training_range": violations,
        "nearest_historical_team": {
            "team": str(nearest["team"]),
            "team_abbreviation": str(nearest["team_abbreviation"]),
            "season": str(nearest["season"]),
        },
    }


def roster_talent_summary(
    embedding: dict[str, float] | pd.Series,
    dataset: pd.DataFrame,
) -> dict[str, float]:
    """Return a descriptive roster-talent index and empirical percentile.

    This is deliberately not a win model. It combines rotation-wide and top-three
    player quality so an unprecedented roster can still be described without
    pretending that historical data supports a Net Rating forecast.
    """

    score = float(
        TALENT_ROTATION_WEIGHT * float(embedding["rotation_quality"])
        + TALENT_TOP_THREE_WEIGHT * float(embedding["top_three_quality"])
    )
    historical = (
        TALENT_ROTATION_WEIGHT
        * pd.to_numeric(dataset["rotation_quality"], errors="coerce")
        + TALENT_TOP_THREE_WEIGHT
        * pd.to_numeric(dataset["top_three_quality"], errors="coerce")
    ).dropna()
    percentile = float(100 * (historical <= score).mean())
    return {
        "talent_index": score,
        "historical_percentile": percentile,
        "historical_min": float(historical.min()),
        "historical_max": float(historical.max()),
    }


def anchor_prediction_to_reference(
    simulated_prediction: dict[str, float],
    baseline_model_prediction: dict[str, float],
    *,
    observed_net_rating: float,
    observed_wins: float,
    observed_games: float,
    win_calibrator: Any,
    net_rating_error: float,
) -> dict[str, float]:
    """Residual-anchor a small roster edit to its team's observed baseline.

    This correction removes the model's baseline shrinkage for a known team and
    applies only the model-estimated change. It is inappropriate for a roster
    built from scratch because that roster has no observed baseline.
    """

    delta_net = float(
        simulated_prediction["net_rating"] - baseline_model_prediction["net_rating"]
    )
    adjusted_net = float(observed_net_rating) + delta_net
    baseline_win_pace = 82 * float(observed_wins) / float(observed_games)
    model_win_delta = float(
        simulated_prediction["expected_wins"]
        - baseline_model_prediction["expected_wins"]
    )
    adjusted_wins = baseline_win_pace + model_win_delta
    win_error = abs(
        float(
            win_calibrator.predict(
                pd.DataFrame(
                    {
                        "net_rating": [
                            adjusted_net + float(net_rating_error),
                            adjusted_net - float(net_rating_error),
                        ]
                    }
                )
            )[0]
            - win_calibrator.predict(
                pd.DataFrame({"net_rating": [adjusted_net]})
            )[0]
        )
    )
    return {
        "net_rating": adjusted_net,
        "net_rating_low": adjusted_net - float(net_rating_error),
        "net_rating_high": adjusted_net + float(net_rating_error),
        "expected_wins": float(np.clip(adjusted_wins, 0, 82)),
        "wins_low": float(np.clip(adjusted_wins - win_error, 0, 82)),
        "wins_high": float(np.clip(adjusted_wins + win_error, 0, 82)),
        "model_delta_net_rating": delta_net,
    }


def profile_scores(
    embedding: dict[str, float] | pd.Series,
    roster: pd.DataFrame | None = None,
    *,
    minute_column: str = "normalized_minutes",
    comparison_pool: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Translate roster dimensions into stable basketball profile categories.

    Frontcourt-only labels use forwards and centers when roster rows are supplied.
    When the league pool is supplied, their traits are re-ranked against other
    frontcourt players so rebounds and blocks are not inflated by comparison with guards.
    """

    rows = []
    for label, traits in PROFILE_CATEGORIES.items():
        use_frontcourt = label in {
            "Frontcourt physicality",
            "Interior defensive tools",
        }
        frontcourt = pd.DataFrame()
        if use_frontcourt and roster is not None and not roster.empty:
            primary_positions = (
                roster["position"].fillna("").astype(str).str.split("-").str[0]
            )
            frontcourt = roster.loc[primary_positions.isin(["F", "C"])].copy()
        if not frontcourt.empty and minute_column in frontcourt:
            weights = pd.to_numeric(
                frontcourt[minute_column], errors="coerce"
            ).fillna(0).clip(lower=0)
            if weights.sum() > 0:
                calibration = pd.DataFrame()
                if comparison_pool is not None and not comparison_pool.empty:
                    comparison_positions = (
                        comparison_pool["position"]
                        .fillna("")
                        .astype(str)
                        .str.split("-")
                        .str[0]
                    )
                    calibration = comparison_pool.loc[
                        comparison_positions.isin(["F", "C"])
                    ]
                trait_scores = []
                for trait in traits:
                    values = pd.to_numeric(
                        frontcourt[f"trait__{trait}"], errors="coerce"
                    ).fillna(0.5)
                    if not calibration.empty:
                        population = np.sort(
                            pd.to_numeric(
                                calibration[f"trait__{trait}"], errors="coerce"
                            ).dropna().to_numpy(dtype=float)
                        )
                        if len(population):
                            left = np.searchsorted(population, values, side="left")
                            right = np.searchsorted(population, values, side="right")
                            values = pd.Series(
                                (left + right + 1) / (2 * len(population)),
                                index=values.index,
                            )
                    trait_scores.append(float(np.average(values, weights=weights)))
                score = float(np.mean(trait_scores))
            else:
                score = float(
                    np.mean([embedding[f"roster__{trait}"] for trait in traits])
                )
        else:
            score = float(
                np.mean([embedding[f"roster__{trait}"] for trait in traits])
            )
        rows.append({"profile": label, "score": score})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def nearest_historical_teams(
    embedding: dict[str, float] | pd.Series,
    dataset: pd.DataFrame,
    *,
    top_k: int = 5,
    exclude: tuple[str, int] | None = None,
) -> pd.DataFrame:
    """Retrieve nearest team-seasons in standardized roster-vector space."""

    matrix = dataset[MODEL_FEATURES].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(imputer.fit_transform(matrix))
    query = pd.DataFrame([{feature: float(embedding[feature]) for feature in MODEL_FEATURES}])
    query_standardized = scaler.transform(imputer.transform(query))[0]
    distances = np.sqrt(np.square(standardized - query_standardized).sum(axis=1))
    result = dataset[
        ["season", "team_id", "team", "team_abbreviation", "wins", "games", "net_rating", "league_rank"]
    ].copy()
    result["embedding_distance"] = distances
    if exclude is not None:
        season, team_id = exclude
        result = result.loc[
            ~(result["season"].eq(season) & result["team_id"].eq(int(team_id)))
        ]
    result["similarity"] = 1 / (1 + result["embedding_distance"])
    return result.nsmallest(top_k, "embedding_distance").reset_index(drop=True)


def primary_team_player_pool(player_team_vectors: pd.DataFrame, season: str) -> pd.DataFrame:
    """Return one current-team row per player using the latest observed team stint."""

    frame = player_team_vectors.loc[player_team_vectors["season"].eq(season)].copy()
    if frame.empty:
        return frame
    return (
        frame.sort_values(["player_id", "last_game_date", "team_minutes"])
        .groupby("player_id", as_index=False)
        .tail(1)
        .sort_values(["team", "player_name"])
        .reset_index(drop=True)
    )


def _fit_rotation_minutes(
    source_minutes: pd.Series,
    *,
    total_minutes: float = 240.0,
    player_cap: float = 48.0,
) -> pd.Series:
    """Convert positive role weights into one legal game without erasing players."""

    minutes = pd.to_numeric(source_minutes, errors="coerce").fillna(0).clip(lower=0)
    if len(minutes) * player_cap < total_minutes:
        raise ValueError("A 240-minute roster requires at least five players.")
    if minutes.sum() <= 0:
        raise ValueError("Roster contribution weights must sum to more than zero.")
    result = pd.Series(0.0, index=minutes.index)
    remaining = float(total_minutes)
    eligible = minutes.gt(0)
    while remaining > 1e-9 and eligible.any():
        weights = minutes.loc[eligible]
        proposed = remaining * weights / weights.sum()
        capped = proposed.ge(player_cap)
        if not capped.any():
            result.loc[eligible] = proposed
            break
        capped_indices = proposed.loc[capped].index
        result.loc[capped_indices] = player_cap
        remaining -= player_cap * len(capped_indices)
        eligible.loc[capped_indices] = False
    return result


def assemble_simulated_roster(
    player_pool: pd.DataFrame,
    player_ids: Iterable[int],
    minute_overrides: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Assemble selected players and fit their observed roles into 240 minutes.

    Player traits are already rate statistics. Observed MPG determines role;
    games played remains reliability metadata. Every roster is fitted to 240
    team minutes with a hard 48-minute player cap.
    """

    ids = list(dict.fromkeys(int(player_id) for player_id in player_ids))
    roster = player_pool.loc[player_pool["player_id"].isin(ids)].copy()
    if len(roster) != len(ids):
        missing = sorted(set(ids) - set(roster["player_id"].astype(int)))
        raise ValueError(f"Unknown player IDs for this season: {missing}")
    overrides = minute_overrides or {}
    source_minutes = roster.apply(
        lambda row: float(
            overrides.get(int(row["player_id"]), row["minutes_per_game"])
        ),
        axis=1,
    ).clip(lower=0)
    if source_minutes.sum() <= 0:
        raise ValueError("Roster contribution weights must sum to more than zero.")
    roster["normalized_minutes"] = _fit_rotation_minutes(source_minutes)
    roster["rotation_weight"] = source_minutes
    roster["model_rotation_share"] = roster["normalized_minutes"] / 240
    return roster.sort_values("normalized_minutes", ascending=False).reset_index(drop=True)


def allocate_reference_rotation(
    reference_roster: pd.DataFrame,
    player_pool: pd.DataFrame,
    player_ids: Iterable[int],
    *,
    total_minutes: float = 240.0,
    forced_player_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Automatically allocate a reference-team counterfactual rotation.

    Incumbents begin with their observed share of 240 team minutes. Added players
    can use minutes opened by removals, then compete with incumbents at overlapping
    positions. Their imported MPG is a ceiling rather than an entitlement: only
    players with lower modeled quality yield minutes, and close-quality players
    yield only a fraction. This compresses weak-player roles on strong teams.
    A caller may explicitly force an addition into the rotation; that player then
    takes overlapping minutes even when the incumbents rate higher. The returned
    frame stores a human-readable allocation audit in ``DataFrame.attrs``.
    """

    ids = list(dict.fromkeys(int(player_id) for player_id in player_ids))
    forced_ids = {int(player_id) for player_id in (forced_player_ids or [])}
    reference_ids = set(reference_roster["player_id"].astype(int))
    selected_reference = reference_roster.loc[
        reference_roster["player_id"].astype(int).isin(ids)
    ].copy()
    added_ids = [player_id for player_id in ids if player_id not in reference_ids]

    # Use the share of minutes each incumbent actually contributed over the
    # season. This handles injuries and trades without pretending that every
    # player's independently observed MPG occurred in the same game.
    source_role = pd.to_numeric(
        reference_roster.get("team_minutes"), errors="coerce"
    ).fillna(0)
    source_role = source_role.where(source_role.gt(0), reference_roster["minutes_per_game"])
    fitted_reference_minutes = _fit_rotation_minutes(source_role)
    minutes = {
        int(reference_roster.loc[index, "player_id"]): float(value)
        for index, value in fitted_reference_minutes.items()
        if int(reference_roster.loc[index, "player_id"]) in ids
    }

    selected_players = player_pool.loc[
        player_pool["player_id"].astype(int).isin(ids)
    ].copy()
    selected_by_id = {
        int(row.player_id): row for row in selected_players.itertuples()
    }

    def audit_impact(row: Any) -> float | None:
        """Return a strict-JSON impact value for dataframe metadata."""

        value = getattr(row, "impact_rating", None)
        return float(value) if value is not None and pd.notna(value) else None

    open_minutes = max(0.0, float(total_minutes - sum(minutes.values())))
    allocation_audit: list[dict[str, Any]] = []
    # Allocate the strongest newcomers first so the same final roster produces
    # the same rotation regardless of the order in which the UI queued players.
    added_ids = sorted(
        added_ids,
        key=lambda player_id: float(selected_by_id[player_id].quality_proxy),
        reverse=True,
    )
    for added_id in added_ids:
        added = selected_by_id[added_id]
        is_forced = added_id in forced_ids
        desired_minutes = float(np.clip(float(added.minutes_per_game), 0.0, 42.0))
        assigned = min(desired_minutes, open_minutes)
        open_minutes_used = assigned
        minutes[added_id] = assigned
        open_minutes -= assigned
        remaining = desired_minutes - assigned
        added_positions = {str(added.position).split("-")[0]}
        added_quality = float(added.quality_proxy)
        displacements: list[dict[str, Any]] = []

        def displacement_key(row: Any) -> tuple[int, float]:
            positions = set(str(row.position).split("-"))
            if positions == added_positions:
                position_priority = 0
            elif positions & added_positions:
                position_priority = 1
            else:
                position_priority = 2
            player_id = int(row.player_id)
            rotation_value = float(row.quality_proxy)
            modeled_rotation_value = float(minutes.get(player_id, 0.0)) * (
                0.5 + rotation_value
            )
            return position_priority, modeled_rotation_value

        competitors = sorted(
            (
                row
                for row in selected_players.itertuples()
                if int(row.player_id) != added_id
                and float(minutes.get(int(row.player_id), 0.0)) > 0
                and bool(set(str(row.position).split("-")) & added_positions)
                and (
                    is_forced
                    or float(row.quality_proxy) < added_quality
                )
            ),
            key=displacement_key,
        )
        for incumbent in competitors:
            incumbent_id = int(incumbent.player_id)
            incumbent_minutes = float(minutes.get(incumbent_id, 0.0))
            incumbent_quality = float(incumbent.quality_proxy)
            quality_advantage = added_quality - incumbent_quality
            replaceable_fraction = (
                1.0
                if is_forced
                else float(np.clip(quality_advantage / 0.20, 0.0, 1.0))
            )
            reduction = min(
                remaining,
                incumbent_minutes * replaceable_fraction,
            )
            minutes[incumbent_id] = incumbent_minutes - reduction
            minutes[added_id] += reduction
            remaining -= reduction
            if reduction > 1e-9:
                displacements.append(
                    {
                        "player_id": incumbent_id,
                        "player_name": str(incumbent.player_name),
                        "position": str(incumbent.position),
                        "minutes": float(reduction),
                        "rotation_value": incumbent_quality,
                        "impact_rating": audit_impact(incumbent),
                    }
                )
            if remaining <= 1e-9:
                break

        blocked_by = sorted(
            [
                {
                    "player_id": int(row.player_id),
                    "player_name": str(row.player_name),
                    "position": str(row.position),
                    "rotation_value": float(row.quality_proxy),
                    "impact_rating": audit_impact(row),
                    "current_minutes": float(minutes.get(int(row.player_id), 0.0)),
                }
                for row in selected_players.itertuples()
                if int(row.player_id) != added_id
                and float(minutes.get(int(row.player_id), 0.0)) > 0
                and bool(set(str(row.position).split("-")) & added_positions)
                and float(row.quality_proxy) >= added_quality
            ],
            key=lambda item: (item["rotation_value"], item["current_minutes"]),
        )
        allocation_audit.append(
            {
                "player_id": added_id,
                "player_name": str(added.player_name),
                "position": str(added.position),
                "rotation_value": added_quality,
                "impact_rating": audit_impact(added),
                "requested_minutes": desired_minutes,
                "open_minutes_used": open_minutes_used,
                "forced": is_forced,
                "displacements": displacements,
                "blocked_by": blocked_by,
            }
        )

    deficit = float(total_minutes - sum(minutes.values()))
    if deficit > 1e-9:
        selected = player_pool.loc[player_pool["player_id"].astype(int).isin(ids)].copy()
        selected["allocation_weight"] = (
            pd.to_numeric(selected["minutes_per_game"], errors="coerce").fillna(0)
            * (
                0.5
                + pd.to_numeric(selected["quality_proxy"], errors="coerce").fillna(0.5)
            )
        )
        eligible = selected.loc[selected["allocation_weight"].gt(0)]
        while deficit > 1e-6 and not eligible.empty:
            weights = eligible["allocation_weight"] / eligible["allocation_weight"].sum()
            distributed = 0.0
            for row, share in zip(eligible.itertuples(), weights, strict=False):
                player_id = int(row.player_id)
                minute_ceiling = (
                    float(np.clip(float(row.minutes_per_game), 0.0, 42.0))
                    if player_id in added_ids
                    else 42.0
                )
                capacity = max(
                    0.0, minute_ceiling - float(minutes.get(player_id, 0.0))
                )
                addition = min(capacity, deficit * float(share))
                minutes[player_id] = float(minutes.get(player_id, 0.0)) + addition
                distributed += addition
            deficit -= distributed
            eligible = eligible.loc[
                eligible.apply(
                    lambda row: float(minutes.get(int(row["player_id"]), 0.0))
                    < (
                        float(np.clip(float(row["minutes_per_game"]), 0.0, 42.0))
                        if int(row["player_id"]) in added_ids
                        else 42.0
                    )
                    - 1e-9,
                    axis=1,
                )
            ]
            if distributed <= 1e-9:
                break

    result = assemble_simulated_roster(player_pool, ids, minutes)
    for item in allocation_audit:
        item["assigned_minutes"] = float(minutes.get(item["player_id"], 0.0))
        item["unfilled_requested_minutes"] = max(
            0.0, item["requested_minutes"] - item["assigned_minutes"]
        )
    result.attrs["allocation_audit"] = allocation_audit
    return result


def embedding_changes(
    baseline: dict[str, float] | pd.Series,
    simulated: dict[str, float] | pd.Series,
) -> pd.DataFrame:
    """Explain the largest trait changes caused by a roster edit."""

    rows = [
        {
            "trait": PLAYER_TRAIT_LABELS[trait],
            "baseline": float(baseline[f"roster__{trait}"]),
            "simulated": float(simulated[f"roster__{trait}"]),
            "change": float(simulated[f"roster__{trait}"] - baseline[f"roster__{trait}"]),
        }
        for trait in PLAYER_TRAITS
    ]
    frame = pd.DataFrame(rows)
    frame["absolute_change"] = frame["change"].abs()
    return frame.sort_values("absolute_change", ascending=False).reset_index(drop=True)
