"""Self-supervised player-style features, embeddings, and retrieval."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler, normalize

from ball_ai.analytics.shot_profile import engineer_shot_events


OFFENSIVE_ACTION_FEATURES = [
    "assisted_make_rate",
    "fastbreak_shot_rate",
    "second_chance_shot_rate",
    "turnover_created_shot_rate",
]

BROAD_OFFENSIVE_FEATURES = [
    "rim_frequency",
    "paint_frequency",
    "midrange_frequency",
    "three_point_frequency",
    "dunk_frequency",
    "layup_frequency",
    "floater_frequency",
    "hook_frequency",
    "pull_up_frequency",
    "step_back_frequency",
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

OFFENSIVE_FEATURES = [*BROAD_OFFENSIVE_FEATURES, *OFFENSIVE_ACTION_FEATURES]
# Best chronological-validation weight under the product constraint that the learned encoder stays primary.
OFFENSIVE_PRESENCE_WEIGHT = 0.4
SHARED_ABSENCE_SHARE = 0.05
ABSENCE_THRESHOLD = 0.01
TEMPORAL_ENSEMBLE_WEIGHT = 0.3
SIAMESE_ENSEMBLE_WEIGHT = 0.4

DEFENSIVE_FEATURES = [
    "defensive_rebounds_per_36",
    "steals_per_36",
    "blocks_per_36",
    "matchup_guard_share",
    "matchup_forward_share",
    "matchup_center_share",
    "matchup_fga_rate",
    "matchup_three_point_rate",
    "matchup_turnover_rate",
    "matchup_block_rate",
    "matchup_shooting_foul_rate",
]

STYLE_FEATURES = {
    "Offensive": OFFENSIVE_FEATURES,
    "Defensive": DEFENSIVE_FEATURES,
    "Overall": [*OFFENSIVE_FEATURES, *DEFENSIVE_FEATURES],
}

STYLE_FEATURE_SETS = {
    "Broad history": {
        "Offensive": BROAD_OFFENSIVE_FEATURES,
        "Defensive": DEFENSIVE_FEATURES,
        "Overall": [*BROAD_OFFENSIVE_FEATURES, *DEFENSIVE_FEATURES],
    },
    "Modern detailed": STYLE_FEATURES,
}

PROFILE_ELIGIBILITY = {
    "Broad history": {
        "Offensive": ("offensive_eligible", "offensive_reliability"),
        "Defensive": ("defensive_eligible", "defensive_reliability"),
        "Overall": ("overall_eligible", "overall_reliability"),
    },
    "Modern detailed": {
        "Offensive": ("modern_offensive_eligible", "modern_offensive_reliability"),
        "Defensive": ("modern_defensive_eligible", "defensive_reliability"),
        "Overall": ("modern_overall_eligible", "modern_overall_reliability"),
    },
}

FEATURE_LABELS = {
    "rim_frequency": "Rim shot frequency",
    "paint_frequency": "Paint shot frequency",
    "midrange_frequency": "Midrange frequency",
    "three_point_frequency": "Three-point frequency",
    "dunk_frequency": "Dunk frequency",
    "layup_frequency": "Layup frequency",
    "floater_frequency": "Floater frequency",
    "hook_frequency": "Hook frequency",
    "pull_up_frequency": "Pull-up frequency",
    "step_back_frequency": "Step-back frequency",
    "average_shot_distance": "Average shot distance",
    "field_goal_attempts_per_36": "FGA per 36",
    "three_point_attempt_rate": "3PA / FGA",
    "free_throw_attempt_rate": "FTA / FGA",
    "assists_per_36": "Assists per 36",
    "turnovers_per_36": "Turnovers per 36",
    "assist_turnover_ratio": "Assist / turnover ratio",
    "offensive_rebounds_per_36": "Offensive rebounds per 36",
    "estimated_used_possessions_per_36": "Estimated used possessions per 36",
    "assisted_make_rate": "Assisted field-goal share",
    "fastbreak_shot_rate": "Fast-break shot share",
    "second_chance_shot_rate": "Second-chance shot share",
    "turnover_created_shot_rate": "Shots created from turnovers",
    "early_clock_shot_rate": "Early-clock shot share",
    "late_clock_shot_rate": "Late-clock shot share",
    "bad_pass_turnover_share": "Bad-pass turnover share",
    "lost_ball_turnover_share": "Lost-ball turnover share",
    "defensive_rebounds_per_36": "Defensive rebounds per 36",
    "steals_per_36": "Steals per 36",
    "blocks_per_36": "Blocks per 36",
    "height_inches": "Height",
    "weight_lbs": "Weight",
    "matchup_guard_share": "Guard matchup share",
    "matchup_forward_share": "Forward matchup share",
    "matchup_center_share": "Center matchup share",
    "matchup_opponent_height": "Average matchup height",
    "matchup_fga_rate": "Matchup FGA per partial possession",
    "matchup_three_point_rate": "Matchup 3PA / FGA",
    "matchup_turnover_rate": "Matchup turnovers per partial possession",
    "matchup_block_rate": "Matchup blocks per FGA",
    "matchup_shooting_foul_rate": "Shooting fouls per partial possession",
}


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


def season_label(start_year: int) -> str:
    """Convert a season start year to the application's season label."""

    return f"{int(start_year)}-{str(int(start_year) + 1)[-2:]}"


def shot_sample_reliability(shot_attempts: float | int | None) -> str:
    """Label retrieval stability using the bootstrap-tested attempt bands."""

    attempts = pd.to_numeric(pd.Series([shot_attempts]), errors="coerce").iloc[0]
    if pd.isna(attempts):
        return "Unavailable"
    if attempts >= 800:
        return "High"
    if attempts >= 400:
        return "Medium"
    return "Limited"


def _zone_mix(frame: pd.DataFrame, group_columns: list[str], prefix: str) -> pd.DataFrame:
    counts = (
        frame.groupby([*group_columns, "zone_bucket"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    totals = counts.sum(axis=1).replace(0, np.nan)
    output = pd.DataFrame(index=counts.index)
    for zone in ("rim", "paint", "midrange", "three_point"):
        output[f"{prefix}_{zone}_frequency"] = counts.get(zone, 0) / totals
    return output.reset_index()


def _mix_shift(frame: pd.DataFrame, left: str, right: str, name: str) -> pd.Series:
    differences = []
    for zone in ("rim", "paint", "midrange", "three_point"):
        differences.append(
            (frame[f"{left}_{zone}_frequency"] - frame[f"{right}_{zone}_frequency"]).abs()
        )
    return pd.concat(differences, axis=1).sum(axis=1).rename(name)


def build_shot_context_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Build home/away, game-phase, and playoff shot-context features.

    Split comparisons require at least 40 shots on each side. Playoff comparisons
    require 100 regular-season and 30 playoff attempts. Missing context remains
    null rather than being encoded as a real zero.
    """

    rename = {
        "PLAYER_ID": "player_id",
        "SHOT_MADE_FLAG": "shot_made",
        "SHOT_DISTANCE": "shot_distance",
        "ACTION_TYPE": "action_type",
        "SHOT_ZONE_BASIC": "shot_zone_basic",
        "TEAM_ID": "team_id",
        "PERIOD": "period",
        "HTM": "home_team",
        "_season": "season_start",
        "_season_type": "season_type",
    }
    frame = shots.rename(columns=rename).copy()
    required = {
        "player_id", "shot_made", "shot_distance", "action_type",
        "shot_zone_basic", "team_id", "period", "home_team",
        "season_start", "season_type",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Shot context rows are missing columns: {sorted(missing)}")

    frame = engineer_shot_events(frame)
    frame["season"] = frame["season_start"].map(season_label)
    team_codes = {
        1610612737: "ATL", 1610612738: "BOS", 1610612751: "BKN",
        1610612766: "CHA", 1610612741: "CHI", 1610612739: "CLE",
        1610612742: "DAL", 1610612743: "DEN", 1610612765: "DET",
        1610612744: "GSW", 1610612745: "HOU", 1610612754: "IND",
        1610612746: "LAC", 1610612747: "LAL", 1610612763: "MEM",
        1610612748: "MIA", 1610612749: "MIL", 1610612750: "MIN",
        1610612740: "NOP", 1610612752: "NYK", 1610612760: "OKC",
        1610612753: "ORL", 1610612755: "PHI", 1610612756: "PHX",
        1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS",
        1610612761: "TOR", 1610612762: "UTA", 1610612764: "WAS",
    }
    frame["location"] = np.where(
        frame["team_id"].map(team_codes).eq(frame["home_team"]), "home", "away"
    )
    frame["game_phase"] = np.select(
        [frame["period"].le(2), frame["period"].eq(4)],
        ["first_half", "fourth_quarter"],
        default="other",
    )

    regular = frame.loc[frame["season_type"].eq("rg")].copy()
    keys = ["player_id", "season"]
    totals = regular.groupby(keys).size().rename("regular_season_shots").reset_index()

    location_counts = (
        regular.groupby([*keys, "location"]).size().unstack(fill_value=0).reset_index()
    )
    for column in ("home", "away"):
        if column not in location_counts:
            location_counts[column] = 0
    location_mix = _zone_mix(regular, [*keys, "location"], "location")
    location_mix = location_mix.pivot(index=keys, columns="location")
    location_mix.columns = [f"{location}_{column.removeprefix('location_')}" for column, location in location_mix.columns]
    location_mix = location_mix.reset_index()

    phases = regular.loc[regular["game_phase"].isin(["first_half", "fourth_quarter"])]
    phase_counts = phases.groupby([*keys, "game_phase"]).size().unstack(fill_value=0).reset_index()
    for column in ("first_half", "fourth_quarter"):
        if column not in phase_counts:
            phase_counts[column] = 0
    phase_mix = _zone_mix(phases, [*keys, "game_phase"], "phase")
    phase_mix = phase_mix.pivot(index=keys, columns="game_phase")
    phase_mix.columns = [f"{phase}_{column.removeprefix('phase_')}" for column, phase in phase_mix.columns]
    phase_mix = phase_mix.reset_index()

    output = totals.merge(location_counts, on=keys, how="left").merge(
        location_mix, on=keys, how="left"
    ).merge(phase_counts, on=keys, how="left").merge(phase_mix, on=keys, how="left")
    output["home_shot_frequency"] = _safe_divide(output["home"], output["regular_season_shots"])
    output["home_away_context_eligible"] = output[["home", "away"]].min(axis=1).ge(40)
    output["quarter_context_eligible"] = output[["first_half", "fourth_quarter"]].min(axis=1).ge(40)
    output["home_away_zone_shift"] = _mix_shift(output, "home", "away", "home_away_zone_shift")
    output["first_half_fourth_quarter_zone_shift"] = _mix_shift(
        output, "first_half", "fourth_quarter", "first_half_fourth_quarter_zone_shift"
    )
    output.loc[~output["home_away_context_eligible"], "home_away_zone_shift"] = np.nan
    output.loc[
        ~output["quarter_context_eligible"], "first_half_fourth_quarter_zone_shift"
    ] = np.nan

    playoffs = frame.loc[frame["season_type"].eq("po")].copy()
    if not playoffs.empty:
        regular_mix = _zone_mix(regular, keys, "regular")
        playoff_mix = _zone_mix(playoffs, keys, "playoff")
        playoff_counts = playoffs.groupby(keys).size().rename("playoff_shots").reset_index()
        postseason = regular_mix.merge(playoff_mix, on=keys, how="inner").merge(
            playoff_counts, on=keys, how="left"
        )
        postseason["playoff_context_eligible"] = (
            postseason["playoff_shots"].ge(30)
        )
        postseason["regular_playoff_zone_shift"] = _mix_shift(
            postseason, "regular", "playoff", "regular_playoff_zone_shift"
        )
        output = output.merge(
            postseason[[*keys, "playoff_shots", "playoff_context_eligible", "regular_playoff_zone_shift"]],
            on=keys,
            how="left",
        )
    else:
        output["playoff_shots"] = np.nan
        output["playoff_context_eligible"] = False
        output["regular_playoff_zone_shift"] = np.nan

    output["playoff_context_eligible"] = (
        output["playoff_context_eligible"].astype("boolean").fillna(False).astype(bool)
    )
    output["playoff_context_eligible"] &= output["regular_season_shots"].ge(100)
    output.loc[
        ~output["playoff_context_eligible"], "regular_playoff_zone_shift"
    ] = np.nan
    return output.drop(columns=["home", "away", "first_half", "fourth_quarter"])


def build_defensive_matchup_profiles(
    matchups: pd.DataFrame,
    biographies: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate NBA partial-matchup rows into defensive behavior profiles."""

    required = {
        "matchups_person_id", "person_id", "position", "game_id",
        "partial_possessions", "matchup_field_goals_attempted",
        "matchup_three_pointers_attempted", "matchup_turnovers",
        "matchup_blocks", "shooting_fouls", "_season", "_season_type",
    }
    if missing := required - set(matchups.columns):
        raise ValueError(f"Matchup rows are missing columns: {sorted(missing)}")
    frame = matchups.loc[matchups["_season_type"].eq("rg")].copy()
    frame["season"] = frame["_season"].map(season_label)
    frame["opponent_position"] = frame["position"].astype("string").str.upper()
    frame["opponent_height_inches"] = np.nan
    if biographies is not None and not biographies.empty:
        bio = biographies.set_index("player_id")
        fallback = pd.Series("other", index=bio.index, dtype="string")
        fallback.loc[pd.to_numeric(bio.get("forward"), errors="coerce").fillna(0).eq(1)] = "F"
        fallback.loc[pd.to_numeric(bio.get("center"), errors="coerce").fillna(0).eq(1)] = "C"
        fallback.loc[pd.to_numeric(bio.get("guard"), errors="coerce").fillna(0).eq(1)] = "G"
        frame["opponent_position"] = frame["opponent_position"].fillna(
            frame["person_id"].map(fallback)
        )
        frame["opponent_height_inches"] = pd.to_numeric(
            frame["person_id"].map(bio.get("heightInches")), errors="coerce"
        )
    frame["opponent_position"] = (
        frame["opponent_position"].str.extract(r"([GFC])", expand=False).fillna("other")
    )
    frame["weighted_position"] = pd.to_numeric(
        frame["partial_possessions"], errors="coerce"
    ).fillna(0)
    frame["weighted_opponent_height"] = (
        frame["opponent_height_inches"] * frame["weighted_position"]
    )

    keys = ["matchups_person_id", "season"]
    totals = frame.groupby(keys).agg(
        matchup_games=("game_id", "nunique"),
        matchup_partial_possessions=("partial_possessions", "sum"),
        matchup_fga=("matchup_field_goals_attempted", "sum"),
        matchup_three_point_attempts=("matchup_three_pointers_attempted", "sum"),
        matchup_turnovers=("matchup_turnovers", "sum"),
        matchup_blocks=("matchup_blocks", "sum"),
        matchup_shooting_fouls=("shooting_fouls", "sum"),
        weighted_opponent_height=("weighted_opponent_height", "sum"),
    ).reset_index().rename(columns={"matchups_person_id": "player_id"})
    positions = (
        frame.groupby([*keys, "opponent_position"], observed=True)["weighted_position"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"matchups_person_id": "player_id"})
    )
    output = totals.merge(positions, on=["player_id", "season"], how="left")
    for position, name in (("G", "guard"), ("F", "forward"), ("C", "center")):
        values = output[position] if position in output else pd.Series(0, index=output.index)
        output[f"matchup_{name}_share"] = _safe_divide(
            values, output["matchup_partial_possessions"]
        ).fillna(0)
    output["matchup_fga_rate"] = _safe_divide(
        output["matchup_fga"], output["matchup_partial_possessions"]
    )
    output["matchup_three_point_rate"] = _safe_divide(
        output["matchup_three_point_attempts"], output["matchup_fga"]
    )
    output["matchup_turnover_rate"] = _safe_divide(
        output["matchup_turnovers"], output["matchup_partial_possessions"]
    )
    output["matchup_block_rate"] = _safe_divide(
        output["matchup_blocks"], output["matchup_fga"]
    )
    output["matchup_shooting_foul_rate"] = _safe_divide(
        output["matchup_shooting_fouls"], output["matchup_partial_possessions"]
    )
    output["matchup_opponent_height"] = _safe_divide(
        output["weighted_opponent_height"], output["matchup_partial_possessions"]
    )
    return output


def build_action_profiles(actions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate identified play-by-play into offensive action tendencies."""

    required = {
        "actionType", "subType", "qualifiers", "personId",
        "assistPersonId", "possession", "orderNumber", "clock", "period",
        "gameId", "shotResult", "_season", "_season_type",
    }
    if missing := required - set(actions.columns):
        raise ValueError(f"Action rows are missing columns: {sorted(missing)}")
    frame = actions.loc[actions["_season_type"].eq("rg")].copy()
    frame = frame.loc[pd.to_numeric(frame["possession"], errors="coerce").gt(0)]
    frame["season"] = frame["_season"].map(season_label)
    frame["action_type"] = frame["actionType"].astype("string").str.lower()
    frame["sub_type"] = frame["subType"].astype("string").str.lower()
    frame["qualifier_text"] = frame["qualifiers"].astype("string").str.lower()
    clock = frame["clock"].astype("string").str.extract(r"PT(?P<minutes>\d+)M(?P<seconds>[\d.]+)S")
    frame["period_seconds"] = (
        pd.to_numeric(clock["minutes"], errors="coerce") * 60
        + pd.to_numeric(clock["seconds"], errors="coerce")
    )
    frame = frame.sort_values(["gameId", "period", "orderNumber"])
    period_group = frame.groupby(["gameId", "period"])
    previous_possession = period_group["possession"].shift()
    previous_seconds = period_group["period_seconds"].shift()
    possession_changed = frame["possession"].ne(previous_possession)
    frame["offensive_rebound"] = frame["action_type"].eq("rebound") & frame["sub_type"].eq("offensive")
    frame["new_sequence"] = possession_changed | frame["offensive_rebound"]
    frame["sequence"] = frame.groupby(["gameId", "period"])["new_sequence"].cumsum()
    sequence_keys = ["gameId", "period", "sequence"]
    frame["sequence_start_marker"] = np.where(
        frame["new_sequence"],
        np.where(
            frame["offensive_rebound"],
            frame["period_seconds"],
            previous_seconds.fillna(frame["period_seconds"]),
        ),
        np.nan,
    )
    frame["sequence_start"] = frame.groupby(sequence_keys)[
        "sequence_start_marker"
    ].transform("max")
    frame["offensive_rebound_sequence"] = frame.groupby(sequence_keys)[
        "offensive_rebound"
    ].transform("max")
    frame["elapsed"] = frame["sequence_start"] - frame["period_seconds"]
    frame["clock_limit"] = np.where(frame["offensive_rebound_sequence"], 14.0, 24.0)
    frame["timing_valid"] = frame["elapsed"].between(0, frame["clock_limit"])

    shots = frame.loc[frame["action_type"].isin(["2pt", "3pt"])].copy()
    shots["made"] = shots["shotResult"].astype("string").str.lower().eq("made")
    shots["assisted_make"] = shots["made"] & shots["assistPersonId"].notna()
    shots["fastbreak"] = shots["qualifier_text"].str.contains("fastbreak", na=False)
    shots["second_chance"] = shots["qualifier_text"].str.contains("2ndchance", na=False)
    shots["from_turnover"] = shots["qualifier_text"].str.contains("fromturnover", na=False)
    shots["early_clock"] = shots["timing_valid"] & shots["elapsed"].le(6)
    shots["late_clock"] = shots["timing_valid"] & (
        shots["clock_limit"] - shots["elapsed"]
    ).le(4)
    keys = ["personId", "season"]
    output = shots.groupby(keys).agg(
        action_shots=("action_type", "size"),
        action_makes=("made", "sum"),
        assisted_makes=("assisted_make", "sum"),
        fastbreak_shots=("fastbreak", "sum"),
        second_chance_shots=("second_chance", "sum"),
        turnover_created_shots=("from_turnover", "sum"),
        timing_shots=("timing_valid", "sum"),
        early_clock_shots=("early_clock", "sum"),
        late_clock_shots=("late_clock", "sum"),
    ).reset_index().rename(columns={"personId": "player_id"})

    turnovers = frame.loc[frame["action_type"].eq("turnover")].copy()
    turnovers["bad_pass"] = turnovers["sub_type"].str.contains("bad pass", na=False)
    turnovers["lost_ball"] = turnovers["sub_type"].str.contains("lost ball", na=False)
    turnover_mix = turnovers.groupby(keys).agg(
        action_turnovers=("action_type", "size"),
        bad_pass_turnovers=("bad_pass", "sum"),
        lost_ball_turnovers=("lost_ball", "sum"),
    ).reset_index().rename(columns={"personId": "player_id"})
    output = output.merge(turnover_mix, on=["player_id", "season"], how="left")
    ratios = {
        "assisted_make_rate": ("assisted_makes", "action_makes"),
        "fastbreak_shot_rate": ("fastbreak_shots", "action_shots"),
        "second_chance_shot_rate": ("second_chance_shots", "action_shots"),
        "turnover_created_shot_rate": ("turnover_created_shots", "action_shots"),
        "early_clock_shot_rate": ("early_clock_shots", "timing_shots"),
        "late_clock_shot_rate": ("late_clock_shots", "timing_shots"),
        "bad_pass_turnover_share": ("bad_pass_turnovers", "action_turnovers"),
        "lost_ball_turnover_share": ("lost_ball_turnovers", "action_turnovers"),
    }
    for target, (numerator, denominator) in ratios.items():
        output[target] = _safe_divide(output[numerator], output[denominator])
    output.loc[output["action_makes"].lt(40), "assisted_make_rate"] = np.nan
    output.loc[output["action_turnovers"].lt(20), [
        "bad_pass_turnover_share", "lost_ball_turnover_share"
    ]] = np.nan
    return output


def _per_36(frame: pd.DataFrame, source: str) -> pd.Series:
    return 36 * _safe_divide(frame[source], frame["minutes_total"])


def build_style_features(
    player_seasons: pd.DataFrame,
    shot_profiles: pd.DataFrame,
    biographies: pd.DataFrame,
    *,
    shot_context: pd.DataFrame | None = None,
    defensive_matchups: pd.DataFrame | None = None,
    action_profiles: pd.DataFrame | None = None,
    impact: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine behavioral inputs and attach explicit eligibility flags."""

    frame = player_seasons.copy()
    frame["minutes_total"] = (
        pd.to_numeric(frame["games_played"], errors="coerce")
        * pd.to_numeric(frame["minutes_per_game"], errors="coerce")
    )
    frame["two_points_made_per_game"] = _safe_divide(
        frame["two_points_made_total"], frame["games_played"]
    )
    frame["two_point_attempts_per_game"] = _safe_divide(
        frame["two_point_attempts_total"], frame["games_played"]
    )
    per_36 = {
        "field_goal_attempts_per_36": "field_goal_attempts_total",
        "assists_per_36": "assists_total",
        "turnovers_per_36": "turnovers_total",
        "offensive_rebounds_per_36": "offensive_rebounds_total",
        "defensive_rebounds_per_36": "defensive_rebounds_total",
        "steals_per_36": "steals_total",
        "blocks_per_36": "blocks_total",
    }
    for target, source in per_36.items():
        frame[target] = _per_36(frame, source)
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
        frame["field_goal_attempts_total"]
        + 0.44 * frame["free_throw_attempts_total"]
        + frame["turnovers_total"]
    )
    frame["estimated_used_possessions_per_36"] = 36 * _safe_divide(
        used, frame["minutes_total"]
    )

    shot_columns = ["player_id", "season", "shot_attempts", *{
        feature for feature in OFFENSIVE_FEATURES if feature in shot_profiles.columns
    }]
    frame = frame.merge(
        shot_profiles[shot_columns].drop_duplicates(["player_id", "season"]),
        on=["player_id", "season"],
        how="left",
        validate="one_to_one",
    )
    biography_columns = [
        column for column in [
            "player_id", "birthDate", "guard", "forward", "center", "heightInches", "bodyWeightLbs"
        ]
        if column in biographies
    ]
    frame = frame.merge(
        biographies[biography_columns].drop_duplicates("player_id"),
        on="player_id",
        how="left",
        validate="many_to_one",
    )
    frame["position"] = frame.apply(
        lambda row: "-".join(
            label for column, label in (("guard", "G"), ("forward", "F"), ("center", "C"))
            if int(row.get(column, 0) or 0) == 1
        ) or "N/A",
        axis=1,
    )
    frame["height_inches"] = pd.to_numeric(frame.get("heightInches"), errors="coerce")
    frame["weight_lbs"] = pd.to_numeric(frame.get("bodyWeightLbs"), errors="coerce")
    birth_dates = pd.to_datetime(frame.get("birthDate"), errors="coerce")
    season_years = pd.to_numeric(frame["season"].str[:4], errors="coerce") + 1
    age_dates = pd.to_datetime(
        season_years.astype("Int64").astype("string") + "-02-01", errors="coerce"
    )
    before_birthday = (
        age_dates.dt.month.lt(birth_dates.dt.month)
        | (
            age_dates.dt.month.eq(birth_dates.dt.month)
            & age_dates.dt.day.lt(birth_dates.dt.day)
        )
    )
    frame["age"] = age_dates.dt.year - birth_dates.dt.year - before_birthday.astype("Int64")
    if shot_context is not None and not shot_context.empty:
        frame = frame.merge(
            shot_context.drop_duplicates(["player_id", "season"]),
            on=["player_id", "season"],
            how="left",
            validate="one_to_one",
        )
    if defensive_matchups is not None and not defensive_matchups.empty:
        frame = frame.merge(
            defensive_matchups.drop_duplicates(["player_id", "season"]),
            on=["player_id", "season"],
            how="left",
            validate="one_to_one",
        )
    if action_profiles is not None and not action_profiles.empty:
        frame = frame.merge(
            action_profiles.drop_duplicates(["player_id", "season"]),
            on=["player_id", "season"],
            how="left",
            validate="one_to_one",
        )
    if impact is not None and not impact.empty:
        frame = frame.merge(
            impact[["player_id", "season", "dpm", "o_dpm", "d_dpm"]].drop_duplicates(
                ["player_id", "season"], keep="last"
            ),
            on=["player_id", "season"],
            how="left",
            validate="one_to_one",
        )

    games = pd.to_numeric(frame["games_played"], errors="coerce").fillna(0)
    minutes = pd.to_numeric(frame["minutes_total"], errors="coerce").fillna(0)
    shots = pd.to_numeric(frame.get("shot_attempts"), errors="coerce").fillna(0)
    matchup_possessions = pd.to_numeric(
        frame.get("matchup_partial_possessions", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).fillna(0)
    matchup_fga = pd.to_numeric(
        frame.get("matchup_fga", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).fillna(0)
    action_shots = pd.to_numeric(
        frame.get("action_shots", pd.Series(np.nan, index=frame.index)), errors="coerce"
    ).fillna(0)
    frame["offensive_eligible"] = (
        games.ge(20) & minutes.ge(500) & shots.ge(100)
    )
    frame["defensive_eligible"] = (
        games.ge(20) & minutes.ge(500) & matchup_possessions.ge(300) & matchup_fga.ge(75)
    )
    frame["overall_eligible"] = frame["offensive_eligible"] & frame["defensive_eligible"]
    frame["modern_offensive_eligible"] = frame["offensive_eligible"] & action_shots.ge(100)
    frame["modern_defensive_eligible"] = frame["defensive_eligible"] & action_shots.ge(100)
    frame["modern_overall_eligible"] = (
        frame["modern_offensive_eligible"] & frame["modern_defensive_eligible"]
    )
    frame["offensive_reliability"] = np.minimum.reduce(
        [games / 50, minutes / 1_500, shots / 500]
    ).clip(0, 1)
    frame["modern_offensive_reliability"] = np.minimum(
        frame["offensive_reliability"], action_shots / 500
    ).clip(0, 1)
    frame["defensive_reliability"] = np.minimum.reduce(
        [games / 50, minutes / 1_500, matchup_possessions / 2_000, matchup_fga / 400]
    ).clip(0, 1)
    frame["overall_reliability"] = frame[[
        "offensive_reliability", "defensive_reliability"
    ]].min(axis=1)
    frame["modern_overall_reliability"] = frame[[
        "modern_offensive_reliability", "defensive_reliability"
    ]].min(axis=1)
    return frame


def _profile_columns(profile: str, lens: str) -> tuple[str, str]:
    try:
        return PROFILE_ELIGIBILITY[profile][lens]
    except KeyError as exc:
        raise ValueError(f"Unknown style profile or lens: {profile} / {lens}") from exc


def _absence_information(features: pd.DataFrame, artifact: dict) -> np.ndarray:
    """Weight shared non-actions by how unusual absence was in training."""

    names = artifact["feature_names"]
    sparse = np.asarray([
        name.endswith("_frequency")
        or name in {"three_point_attempt_rate", "free_throw_attempt_rate"}
        for name in names
    ])
    weights = np.broadcast_to(
        np.asarray(artifact.get("feature_weights", 1.0), dtype=float), len(names)
    )
    training = features
    if "training_end" in artifact and "season" in features:
        training = training.loc[training["season"].le(artifact["training_end"])]
    eligibility = artifact.get("eligibility_column")
    if eligibility in training:
        training = training.loc[training[eligibility].fillna(False)]
    raw = artifact["imputer"].transform(training[names])
    information = np.zeros(len(names))
    information[sparse] = (
        1 - (raw[:, sparse] <= ABSENCE_THRESHOLD).mean(axis=0)
    ) * weights[sparse]
    return information


def _activate(values: np.ndarray, activation: str) -> np.ndarray:
    if activation == "tanh":
        return np.tanh(values)
    if activation == "relu":
        return np.maximum(values, 0)
    raise ValueError(f"Unsupported encoder activation: {activation}")


def fit_style_artifact(
    features: pd.DataFrame,
    lens: str,
    *,
    profile: str = "Modern detailed",
    training_end: str = "2022-23",
    latent_dimensions: int = 12,
    pca_dimensions: int = 12,
    random_state: int = 42,
    feature_weights: dict[str, float] | None = None,
    feature_names: Iterable[str] | None = None,
    max_iter: int = 500,
) -> dict:
    """Fit a denoising autoencoder and matching statistical baselines."""

    columns = list(feature_names or STYLE_FEATURE_SETS[profile][lens])
    eligibility_column, reliability_column = _profile_columns(profile, lens)
    eligible = features[eligibility_column].fillna(False)
    training = features.loc[eligible & features["season"].le(training_end), columns]
    if len(training) < 50:
        raise ValueError(f"Only {len(training)} eligible {lens.lower()} training rows are available.")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    clean = scaler.fit_transform(imputer.fit_transform(training))
    weights = np.asarray([
        (feature_weights or {}).get(column, 1.0) for column in columns
    ])
    clean *= weights
    dimensions = min(latent_dimensions, clean.shape[1], len(clean) - 1)
    pca = PCA(
        n_components=min(pca_dimensions, clean.shape[1], len(clean) - 1),
        random_state=random_state,
    ).fit(clean)

    rng = np.random.default_rng(random_state)
    target = np.tile(clean, (3, 1))
    corrupted = target + rng.normal(0, 0.05, size=target.shape)
    corrupted[rng.random(corrupted.shape) < 0.10] = 0.0
    encoder = MLPRegressor(
        hidden_layer_sizes=(dimensions,),
        activation="tanh",
        alpha=0.001,
        early_stopping=True,
        max_iter=max_iter,
        random_state=random_state,
    ).fit(corrupted, target)
    return {
        "lens": lens,
        "profile": profile,
        "feature_names": columns,
        "eligibility_column": eligibility_column,
        "reliability_column": reliability_column,
        "training_end": training_end,
        "training_rows": len(training),
        "latent_dimensions": dimensions,
        "max_iter": max_iter,
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
        "encoder": encoder,
        "feature_weights": weights,
    }


def fit_temporal_contrastive_artifact(
    features: pd.DataFrame,
    *,
    training_end: str,
    latent_dimensions: int = 16,
    seasons_per_player: int = 3,
    max_iter: int = 30,
    random_state: int = 42,
) -> dict:
    """Fit temporal metric learning without exposing player identity as an input.

    Player identity supplies self-supervised positive classes only: repeated seasons
    from one player are pulled together while other player-seasons act as negatives.
    """

    columns = BROAD_OFFENSIVE_FEATURES
    training = features.loc[
        features["offensive_eligible"].fillna(False)
        & features["season"].le(training_end)
    ].sort_values("season")
    training = training.groupby("player_id", as_index=False).tail(seasons_per_player)
    counts = training["player_id"].value_counts()
    training = training.loc[training["player_id"].isin(counts[counts >= 2].index)]
    if training["player_id"].nunique() < 2:
        raise ValueError("Temporal metric learning needs at least two repeated players.")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    clean = scaler.fit_transform(imputer.fit_transform(training[columns]))
    dimensions = min(latent_dimensions, len(columns))
    encoder = NeighborhoodComponentsAnalysis(
        n_components=dimensions,
        init="pca",
        max_iter=max_iter,
        tol=1e-4,
        random_state=random_state,
    ).fit(clean, training["player_id"].astype(str))
    return {
        "model": "Temporal contrastive NCA",
        "feature_names": columns,
        "training_end": training_end,
        "training_rows": len(training),
        "training_players": int(training["player_id"].nunique()),
        "latent_dimensions": dimensions,
        "seasons_per_player": seasons_per_player,
        "imputer": imputer,
        "scaler": scaler,
        "encoder": encoder,
    }


def transform_temporal_contrastive(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    """Transform player-season behavior into normalized temporal metric space."""

    clean = artifact["scaler"].transform(
        artifact["imputer"].transform(frame[artifact["feature_names"]])
    )
    return normalize(artifact["encoder"].transform(clean))


def transform_style(artifact: dict, frame: pd.DataFrame, method: str) -> np.ndarray:
    """Transform player rows with a fitted baseline or learned encoder."""

    clean = artifact["scaler"].transform(
        artifact["imputer"].transform(frame[artifact["feature_names"]])
    )
    clean *= artifact.get("feature_weights", 1.0)
    if method == "Cosine baseline":
        values = clean
    elif method == "PCA baseline":
        values = artifact["pca"].transform(clean)
    elif method == "Denoising autoencoder":
        encoder = artifact["encoder"]
        values = _activate(
            clean @ encoder.coefs_[0] + encoder.intercepts_[0], encoder.activation
        )
    else:
        raise ValueError(f"Unknown style embedding method: {method}")
    return normalize(values)


def build_embedding_frame(features: pd.DataFrame, artifact: dict, method: str) -> pd.DataFrame:
    values = transform_style(artifact, features, method)
    output = features[[
        "player_id", "player_name", "season", "age", "position", "games_played",
        "shot_attempts", "dpm", "o_dpm", "d_dpm",
        artifact["eligibility_column"], artifact["reliability_column"],
    ]].copy()
    output = output.rename(columns={
        artifact["eligibility_column"]: "eligible",
        artifact["reliability_column"]: "reliability",
    })
    output["profile"] = artifact["profile"]
    output["lens"] = artifact["lens"]
    output["method"] = method
    for index in range(values.shape[1]):
        output[f"embedding_{index:02d}"] = values[:, index]
    return output


def evaluate_temporal_retrieval(
    embeddings: pd.DataFrame,
    *,
    query_season: str,
    candidate_season: str,
) -> dict:
    """Measure whether a season retrieves the same player's prior season."""

    embedding_columns = [column for column in embeddings if column.startswith("embedding_")]
    query = embeddings.loc[
        embeddings["season"].eq(query_season) & embeddings["eligible"].fillna(False)
    ].copy()
    candidates = embeddings.loc[
        embeddings["season"].eq(candidate_season) & embeddings["eligible"].fillna(False)
    ].copy()
    shared = set(query["player_id"]) & set(candidates["player_id"])
    query = query.loc[query["player_id"].isin(shared)].reset_index(drop=True)
    candidates = candidates.reset_index(drop=True)
    if not shared or candidates.empty:
        raise ValueError("No eligible adjacent-season players are available for evaluation.")

    scores = query[embedding_columns].to_numpy() @ candidates[embedding_columns].to_numpy().T
    candidate_ids = candidates["player_id"].to_numpy()
    ranks = []
    for row_index, player_id in enumerate(query["player_id"].to_numpy()):
        order = np.argsort(-scores[row_index])
        rank = int(np.flatnonzero(candidate_ids[order] == player_id)[0]) + 1
        ranks.append(rank)
    ranks_array = np.asarray(ranks)
    return {
        "query_season": query_season,
        "candidate_season": candidate_season,
        "eligible_queries": len(ranks),
        "recall_at_1": float(np.mean(ranks_array <= 1)),
        "recall_at_3": float(np.mean(ranks_array <= 3)),
        "recall_at_5": float(np.mean(ranks_array <= 5)),
        "recall_at_10": float(np.mean(ranks_array <= 10)),
        "mean_reciprocal_rank": float(np.mean(1 / ranks_array)),
        "median_rank": float(np.median(ranks_array)),
    }


def temporal_self_match(
    embeddings: pd.DataFrame,
    player_id: int,
    *,
    lens: str,
    method: str,
    query_season: str,
    candidate_season: str,
    profile: str | None = None,
) -> dict:
    """Return one player's prior-season retrieval rank and similarity."""

    cohort = embeddings.loc[
        embeddings["lens"].eq(lens) & embeddings["method"].eq(method)
    ]
    if profile is not None:
        cohort = cohort.loc[cohort["profile"].eq(profile)]
    eligible = cohort["eligible"].astype("boolean").fillna(False).astype(bool)
    query = cohort.loc[
        cohort["season"].eq(query_season)
        & cohort["player_id"].astype(int).eq(int(player_id))
        & eligible
    ]
    candidates = cohort.loc[
        cohort["season"].eq(candidate_season) & eligible
    ].copy()
    target = candidates.loc[candidates["player_id"].astype(int).eq(int(player_id))]
    if query.empty or target.empty:
        raise ValueError("This player is not eligible in both adjacent seasons.")
    columns = [column for column in cohort if column.startswith("embedding_")]
    scores = candidates[columns].to_numpy() @ query.iloc[0][columns].to_numpy(dtype=float)
    order = np.argsort(-scores)
    target_index = int(np.flatnonzero(candidates.iloc[order]["player_id"].astype(int).eq(int(player_id)))[0])
    cosine = float(scores[order[target_index]])
    return {
        "rank": target_index + 1,
        "candidate_count": len(candidates),
        "cosine_similarity": cosine,
        "similarity_score": float(1 - np.arccos(np.clip(cosine, -1, 1)) / np.pi),
        "query_season": query_season,
        "candidate_season": candidate_season,
    }


def historical_self_similarities(
    embeddings: pd.DataFrame,
    player_id: int,
    *,
    lens: str,
    method: str,
    season: str,
    profile: str | None = None,
    limit: int = 3,
) -> pd.DataFrame:
    """Return recent prior seasons for context without mixing them into player results."""

    cohort = embeddings.loc[
        embeddings["lens"].eq(lens) & embeddings["method"].eq(method)
    ].copy()
    if profile is not None:
        cohort = cohort.loc[cohort["profile"].eq(profile)]
    reference = cohort.loc[
        cohort["player_id"].astype(int).eq(int(player_id))
        & cohort["season"].eq(season)
    ]
    history = cohort.loc[
        cohort["player_id"].astype(int).eq(int(player_id))
        & cohort["season"].lt(season)
        & cohort["eligible"].astype("boolean").fillna(False)
    ].copy()
    if reference.empty or history.empty:
        return history.assign(similarity_score=pd.Series(dtype=float))
    columns = [column for column in cohort if column.startswith("embedding_")]
    cosine = history[columns].to_numpy() @ reference.iloc[0][columns].to_numpy(dtype=float)
    history["similarity_score"] = 1 - np.arccos(np.clip(cosine, -1, 1)) / np.pi
    return history.sort_values("season", ascending=False).head(limit).reset_index(drop=True)


def find_style_neighbors(
    embeddings: pd.DataFrame,
    player_id: int,
    *,
    lens: str,
    method: str,
    season: str,
    profile: str | None = None,
    top_n: int = 5,
    position_aware: bool = False,
    impact_tolerance: float | None = None,
    candidate_season_start: str | None = None,
    candidate_season_end: str | None = None,
    minimum_age: int | None = None,
    maximum_age: int | None = None,
    features: pd.DataFrame | None = None,
    artifact: dict | None = None,
    presence_weight: float = 0.0,
    temporal_artifact: dict | None = None,
    temporal_weight: float = 0.0,
    siamese_embeddings: pd.DataFrame | None = None,
    siamese_weight: float = 0.0,
    unique_players: bool = False,
    candidate_player_id: int | None = None,
) -> pd.DataFrame:
    """Retrieve filtered historical neighbors from a learned or baseline style space."""

    cohort = embeddings.loc[
        embeddings["lens"].eq(lens)
        & embeddings["method"].eq(method)
    ].copy()
    if profile is not None:
        cohort = cohort.loc[cohort["profile"].eq(profile)]
    reference_rows = cohort.loc[
        cohort["player_id"].astype(int).eq(int(player_id))
        & cohort["season"].eq(season)
    ]
    if reference_rows.empty:
        raise ValueError(f"No {lens.lower()} embedding is available for this player and season.")
    reference = reference_rows.iloc[0]
    candidates = cohort.loc[
        cohort["eligible"].astype("boolean").fillna(False).astype(bool)
    ].copy()
    if candidate_season_start is not None:
        candidates = candidates.loc[candidates["season"].ge(candidate_season_start)]
    if candidate_season_end is not None:
        candidates = candidates.loc[candidates["season"].le(candidate_season_end)]
    if minimum_age is not None:
        candidates = candidates.loc[pd.to_numeric(candidates["age"], errors="coerce").ge(minimum_age)]
    if maximum_age is not None:
        candidates = candidates.loc[pd.to_numeric(candidates["age"], errors="coerce").le(maximum_age)]
    if position_aware:
        tokens = set(str(reference["position"]).split("-")) - {"N/A"}
        candidates = candidates.loc[
            candidates["position"].map(lambda value: bool(tokens & set(str(value).split("-"))))
        ]
    if candidate_player_id is None:
        candidates = candidates.loc[
            ~candidates["player_id"].astype(int).eq(int(player_id))
        ]
    else:
        candidates = candidates.loc[
            candidates["player_id"].astype(int).eq(int(candidate_player_id))
            & ~(
                candidates["player_id"].astype(int).eq(int(player_id))
                & candidates["season"].eq(season)
            )
        ]
    impact_column = {"Offensive": "o_dpm", "Defensive": "d_dpm", "Overall": "dpm"}[lens]
    reference_impact = pd.to_numeric(
        pd.Series([reference[impact_column]]), errors="coerce"
    ).iloc[0]
    candidate_impact = pd.to_numeric(candidates[impact_column], errors="coerce")
    candidates["impact_difference"] = candidate_impact - reference_impact
    if impact_tolerance is not None:
        if pd.isna(reference_impact):
            raise ValueError(
                f"Comparable Player impact is unavailable for {season}. Use Style Twin instead."
            )
        candidates = candidates.loc[
            candidate_impact.notna()
            & candidates["impact_difference"].abs().le(impact_tolerance)
        ]
    if candidates.empty:
        raise ValueError("No eligible comparison candidates remain after filtering.")
    columns = [column for column in cohort if column.startswith("embedding_")]
    cosine = candidates[columns].to_numpy() @ reference[columns].to_numpy(dtype=float)
    candidates["cosine_similarity"] = np.clip(cosine, -1, 1)
    candidates["embedding_similarity"] = (
        1 - np.arccos(candidates["cosine_similarity"]) / np.pi
    )
    candidates["positive_behavior_overlap"] = np.nan
    candidates["similarity_score"] = candidates["embedding_similarity"]
    if presence_weight:
        if not 0 <= presence_weight <= 1:
            raise ValueError("Presence weight must be between 0 and 1.")
        if features is None or artifact is None:
            raise ValueError("Presence-aware retrieval requires features and a model artifact.")
        feature_names = artifact["feature_names"]
        indexed = features.drop_duplicates(["player_id", "season"], keep="last").copy()
        indexed["player_id"] = indexed["player_id"].astype(int)
        indexed = indexed.set_index(["player_id", "season"])
        reference_key = (int(player_id), season)
        candidate_keys = pd.MultiIndex.from_frame(
            candidates[["player_id", "season"]].assign(
                player_id=lambda frame: frame["player_id"].astype(int)
            )
        )
        if reference_key not in indexed.index or not candidate_keys.isin(indexed.index).all():
            raise ValueError("Presence-aware retrieval is missing player-season features.")
        raw_reference = artifact["imputer"].transform(
            indexed.loc[[reference_key], feature_names]
        )[0]
        raw_candidates = artifact["imputer"].transform(
            indexed.reindex(candidate_keys)[feature_names]
        )
        scale = np.where(artifact["scaler"].scale_ == 0, 1, artifact["scaler"].scale_)
        weights = artifact.get("feature_weights", 1.0)
        positive_reference = np.clip(raw_reference / scale * weights, 0, None)
        positive_candidates = np.clip(raw_candidates / scale * weights, 0, None)
        overlap = np.minimum(positive_candidates, positive_reference).sum(axis=1)
        union = np.maximum(positive_candidates, positive_reference).sum(axis=1)
        candidates["positive_behavior_overlap"] = np.divide(
            overlap, union, out=np.zeros_like(overlap), where=union != 0
        )
        absence_information = _absence_information(features, artifact)
        reference_absent = raw_reference <= ABSENCE_THRESHOLD
        candidates_absent = raw_candidates <= ABSENCE_THRESHOLD
        shared_absence = (
            (candidates_absent & reference_absent) * absence_information
        ).sum(axis=1)
        either_absent = (
            (candidates_absent | reference_absent) * absence_information
        ).sum(axis=1)
        candidates["shared_absence_overlap"] = np.divide(
            shared_absence,
            either_absent,
            out=np.zeros_like(shared_absence),
            where=either_absent != 0,
        )
        candidates["behavior_overlap"] = (
            (1 - SHARED_ABSENCE_SHARE) * candidates["positive_behavior_overlap"]
            + SHARED_ABSENCE_SHARE * candidates["shared_absence_overlap"]
        )
        candidates["similarity_score"] = (
            (1 - presence_weight) * candidates["embedding_similarity"]
            + presence_weight * candidates["behavior_overlap"]
        )
    candidates["temporal_similarity"] = np.nan
    if temporal_weight:
        if not 0 <= temporal_weight <= 1:
            raise ValueError("Temporal weight must be between 0 and 1.")
        if features is None or temporal_artifact is None:
            raise ValueError("Temporal retrieval requires features and its model artifact.")
        indexed = features.drop_duplicates(["player_id", "season"], keep="last").copy()
        indexed["player_id"] = indexed["player_id"].astype(int)
        indexed = indexed.set_index(["player_id", "season"])
        candidate_keys = pd.MultiIndex.from_frame(
            candidates[["player_id", "season"]].assign(
                player_id=lambda frame: frame["player_id"].astype(int)
            )
        )
        reference_features = indexed.loc[[(int(player_id), season)]].reset_index()
        candidate_features = indexed.reindex(candidate_keys).reset_index()
        temporal_reference = transform_temporal_contrastive(
            temporal_artifact, reference_features
        )[0]
        temporal_candidates = transform_temporal_contrastive(
            temporal_artifact, candidate_features
        )
        temporal_cosine = np.clip(temporal_candidates @ temporal_reference, -1, 1)
        candidates["temporal_similarity"] = 1 - np.arccos(temporal_cosine) / np.pi
        candidates["similarity_score"] = (
            (1 - temporal_weight) * candidates["similarity_score"]
            + temporal_weight * candidates["temporal_similarity"]
        )
    candidates["siamese_similarity"] = np.nan
    if siamese_weight:
        if not 0 <= siamese_weight <= 1:
            raise ValueError("Siamese weight must be between 0 and 1.")
        if siamese_embeddings is None:
            raise ValueError("Siamese retrieval requires precomputed embeddings.")
        columns = [
            column for column in siamese_embeddings
            if column.startswith("siamese_embedding_")
        ]
        indexed = siamese_embeddings.drop_duplicates(
            ["player_id", "season"], keep="last"
        ).copy()
        indexed["player_id"] = indexed["player_id"].astype(int)
        indexed = indexed.set_index(["player_id", "season"])
        reference_key = (int(player_id), season)
        candidate_keys = pd.MultiIndex.from_frame(
            candidates[["player_id", "season"]].assign(
                player_id=lambda frame: frame["player_id"].astype(int)
            )
        )
        if (
            not columns
            or reference_key not in indexed.index
            or not candidate_keys.isin(indexed.index).all()
        ):
            raise ValueError("Siamese retrieval is missing player-season embeddings.")
        reference_vector = indexed.loc[reference_key, columns].to_numpy(dtype=float)
        candidate_vectors = indexed.reindex(candidate_keys)[columns].to_numpy(dtype=float)
        cosine = np.clip(candidate_vectors @ reference_vector, -1, 1)
        candidates["siamese_similarity"] = 1 - np.arccos(cosine) / np.pi
        candidates["similarity_score"] = (
            (1 - siamese_weight) * candidates["similarity_score"]
            + siamese_weight * candidates["siamese_similarity"]
        )
    candidates["dpm_difference"] = candidates["dpm"] - reference["dpm"]
    candidates["o_dpm_difference"] = candidates["o_dpm"] - reference["o_dpm"]
    candidates["d_dpm_difference"] = candidates["d_dpm"] - reference["d_dpm"]
    candidates = candidates.sort_values("similarity_score", ascending=False)
    if unique_players:
        candidates = candidates.drop_duplicates("player_id", keep="first")
    return candidates.head(top_n).reset_index(drop=True)


def input_feature_comparison(
    features: pd.DataFrame,
    artifact: dict,
    reference_id: int,
    candidate_id: int,
    season: str,
    candidate_season: str | None = None,
) -> pd.DataFrame:
    """Explain learned retrieval with standardized observed-input gaps."""

    reference = features.loc[
        features["season"].eq(season)
        & features["player_id"].astype(int).eq(int(reference_id))
    ]
    candidate = features.loc[
        features["season"].eq(candidate_season or season)
        & features["player_id"].astype(int).eq(int(candidate_id))
    ]
    if reference.empty or candidate.empty:
        raise ValueError("Both players need style features in the selected season.")
    columns = artifact["feature_names"]
    raw = pd.concat([reference.iloc[[0]], candidate.iloc[[0]]])[columns]
    scaled = artifact["scaler"].transform(artifact["imputer"].transform(raw))
    weights = artifact.get("feature_weights", 1.0)
    scaled *= weights
    scale = np.where(artifact["scaler"].scale_ == 0, 1, artifact["scaler"].scale_)
    shared_presence = np.minimum(raw.iloc[0], raw.iloc[1]).to_numpy(dtype=float)
    shared_presence = np.clip(shared_presence / scale * weights, 0, None)
    gap = np.abs(scaled[0] - scaled[1])
    absence_information = _absence_information(features, artifact)
    shared_absence = (
        (raw.iloc[0].to_numpy(dtype=float) <= ABSENCE_THRESHOLD)
        & (raw.iloc[1].to_numpy(dtype=float) <= ABSENCE_THRESHOLD)
    ) * absence_information
    return pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(column, column) for column in columns],
        "Reference": raw.iloc[0].to_numpy(),
        "Match": raw.iloc[1].to_numpy(),
        "Standardized gap": gap,
        "Shared tendency strength": shared_presence / (1 + gap),
        "Weak shared-absence evidence": shared_absence,
    }).sort_values("Standardized gap")
