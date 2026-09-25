"""Prepare compact, auditable evidence packets for deterministic explanations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ball_ai.analytics.metrics import summarize_games
from ball_ai.analytics.similarity import FEATURE_LABELS
from ball_ai.analytics.stat_order import BASKETBALL_REFERENCE_STAT_ORDER
from ball_ai.analytics.trends import calculate_trends


SEASON_EVIDENCE = [
    (item.key, f"Season {item.label.lower()}", item.value_format)
    for item in BASKETBALL_REFERENCE_STAT_ORDER
]

SHOT_EVIDENCE = [
    ("shot_attempts", "Season shot attempts", "integer"),
    ("rim_frequency", "Rim attempt frequency", "percentage"),
    ("paint_frequency", "Non-restricted-paint attempt frequency", "percentage"),
    ("midrange_frequency", "Midrange attempt frequency", "percentage"),
    ("three_point_frequency", "Three-point attempt frequency", "percentage"),
    ("dunk_frequency", "Dunk attempt frequency", "percentage"),
    ("layup_frequency", "Layup attempt frequency", "percentage"),
    ("floater_frequency", "Floater attempt frequency", "percentage"),
    ("hook_frequency", "Hook-shot attempt frequency", "percentage"),
    ("pull_up_frequency", "Pull-up attempt frequency", "percentage"),
    ("step_back_frequency", "Step-back attempt frequency", "percentage"),
    ("average_shot_distance", "Average shot distance in feet", "number"),
    ("shot_difficulty_index", "Expected-shot difficulty index", "percentage"),
    ("shot_making_above_expected", "Shot making above expected", "percentage"),
]


def _python_value(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def build_player_evidence(
    profile: dict,
    games: pd.DataFrame,
    data_status: str | None = None,
    season_history: pd.DataFrame | None = None,
) -> dict:
    """Build the only data payload exposed to the report generator."""

    status = data_status or "Bundled offline demo snapshot; recent game rows are illustrative."
    is_real_snapshot = status.startswith("CC0-licensed Kaggle snapshot")
    recent_context = "stored licensed game rows" if is_real_snapshot else "demo games"
    split_context = "stored licensed rows" if is_real_snapshot else "illustrative rows"
    season = {
        key: _python_value(profile[key])
        for key, _, _ in SEASON_EVIDENCE
        if key in profile and pd.notna(profile[key])
    }
    games_played = pd.to_numeric(profile.get("games_played"), errors="coerce")
    attempts_per_game = pd.to_numeric(
        profile.get("three_point_attempts_per_game"), errors="coerce"
    )
    makes_per_game = pd.to_numeric(
        profile.get("three_points_made_per_game"), errors="coerce"
    )
    if (
        pd.isna(games_played)
        or pd.isna(attempts_per_game)
        or pd.isna(makes_per_game)
        or attempts_per_game * games_played < 25
        or makes_per_game * games_played < 5
    ):
        season.pop("three_point_percentage", None)
    shot_profile = {
        key: _python_value(profile[key])
        for key, _, _ in SHOT_EVIDENCE
        if key in profile and pd.notna(profile[key])
    }
    evidence = []
    for index, (key, label, kind) in enumerate(SEASON_EVIDENCE, start=1):
        if key in season:
            evidence.append(
                {
                    "evidence_id": f"E{index}",
                    "metric": key,
                    "label": label,
                    "value": season[key],
                    "format": kind,
                    "context": str(profile.get("season", "season")),
                }
            )

    next_id = len(evidence) + 1
    for key, label, kind in SHOT_EVIDENCE:
        if key in shot_profile:
            evidence.append(
                {
                    "evidence_id": f"E{next_id}",
                    "metric": key,
                    "label": label,
                    "value": shot_profile[key],
                    "format": kind,
                    "context": f"{profile.get('season', 'season')} NBA shot-detail snapshot",
                }
            )
            next_id += 1

    recent = summarize_games(games)
    trends = calculate_trends(games)
    for key in ("effective_field_goal_percentage", "rebounds", "assists", "points"):
        if key in recent:
            evidence.append(
                {
                    "evidence_id": f"E{next_id}",
                    "metric": f"recent_{key}",
                    "label": f"Recent {key.replace('_', ' ')} average",
                    "value": recent[key],
                    "format": "percentage" if "percentage" in key else "number",
                    "context": f"last {recent['games']} {recent_context}",
                }
            )
            next_id += 1

    for trend in trends:
        value_format = "percentage" if "percentage" in trend["metric"] else "number"
        for split, label_prefix in (("prior_average", "Prior"), ("recent_average", "Latest")):
            evidence.append(
                {
                    "evidence_id": f"E{next_id}",
                    "metric": f"trend_{trend['metric']}_{split}",
                    "label": f"{label_prefix} {trend['label']} average",
                    "value": trend[split],
                    "format": value_format,
                    "context": f"{trend['window']}-game split from {split_context}",
                }
            )
            next_id += 1

    career_context: dict[str, Any] = {}
    if season_history is not None and not season_history.empty:
        ordered_history = season_history.sort_values("season")
        current_season = str(profile.get("season", ""))
        prior = ordered_history.loc[ordered_history["season"].astype(str) < current_season]
        career_context = {
            "first_season_available": str(ordered_history.iloc[0]["season"]),
            "latest_season_available": str(ordered_history.iloc[-1]["season"]),
            "seasons_available": int(len(ordered_history)),
        }
        if not prior.empty:
            previous = prior.iloc[-1]
            previous_season = str(previous["season"])
            career_context["previous_season"] = previous_season
            for key, label, kind in (
                ("points_per_game", "Previous-season points per game", "number"),
                ("rebounds_per_game", "Previous-season rebounds per game", "number"),
                ("assists_per_game", "Previous-season assists per game", "number"),
                ("three_point_percentage", "Previous-season three-point percentage", "percentage"),
            ):
                if key == "three_point_percentage":
                    attempts = pd.to_numeric(
                        previous.get("three_point_attempts_total"), errors="coerce"
                    )
                    makes = pd.to_numeric(
                        previous.get("three_points_made_total"), errors="coerce"
                    )
                    if pd.isna(attempts) or pd.isna(makes) or attempts < 25 or makes < 5:
                        continue
                if key in previous and pd.notna(previous[key]):
                    value = _python_value(previous[key])
                    career_context[key] = value
                    evidence.append(
                        {
                            "evidence_id": f"E{next_id}",
                            "metric": f"previous_{key}",
                            "label": label,
                            "value": value,
                            "format": kind,
                            "context": f"{previous_season} historical Parquet archive",
                        }
                    )
                    next_id += 1

    limitations = [
        "The stored CC0 dataset is a point-in-time snapshot and may become stale."
        if is_real_snapshot
        else "The bundled season data is a static demo snapshot, not live data.",
        "Recent games are the latest rows selected from the licensed snapshot."
        if is_real_snapshot
        else "Recent game rows are deterministic illustrative examples, not official box scores.",
        "Shot difficulty is an expected-FG proxy from zone and action mix, not a direct defender-pressure measurement."
        if shot_profile
        else "Shot location and action-type data are not available.",
        "Statistical output does not include film, injuries, lineup context, tracking, or defensive scheme.",
        "Three-point percentage is withheld from the summary below 25 attempts or five makes.",
    ]
    if career_context:
        limitations.append(
            "Career context uses the locally stored regular-season Parquet archive; source-era gaps remain null."
        )
    return {
        "data_status": status,
        "player": {
            "player_id": int(profile["player_id"]),
            "player_name": str(profile["player_name"]),
            "team": str(profile["team"]),
            "position": str(profile["position"]),
            "season": str(profile["season"]),
            "games_played": int(profile["games_played"]),
        },
        "season_stats": season,
        "shot_profile": shot_profile,
        "recent_summary": recent,
        "trends": trends,
        "career_context": career_context,
        "evidence": evidence,
        "limitations": limitations,
    }


def build_comparison_evidence(
    first_profile: dict,
    first_games: pd.DataFrame,
    second_profile: dict,
    second_games: pd.DataFrame,
    data_status: str | None = None,
) -> dict:
    """Combine two independently grounded player packets for comparison."""

    return {
        "comparison_type": "two-player statistical profile",
        "players": [
            build_player_evidence(first_profile, first_games, data_status),
            build_player_evidence(second_profile, second_games, data_status),
        ],
    }


def build_similarity_evidence(
    reference_profile: dict,
    neighbors: pd.DataFrame,
    preset: str,
    position_aware: bool,
    minimum_games: int,
    data_status: str | None = None,
    minimum_shots: int = 0,
) -> dict:
    """Build an auditable packet for explaining deterministic vector retrieval."""

    matches = []
    for match_index, row in enumerate(neighbors.itertuples(index=False), start=1):
        values = row._asdict()
        evidence = []
        active_features = [
            feature
            for feature in FEATURE_LABELS
            if f"contribution__{feature}" in values
        ]
        for feature_index, feature in enumerate(active_features, start=1):
            evidence.append(
                {
                    "evidence_id": f"M{match_index}E{feature_index}",
                    "feature": feature,
                    "label": FEATURE_LABELS[feature],
                    "reference_value": _python_value(reference_profile[feature]),
                    "match_value": _python_value(values[feature]),
                    "reference_percentile": _python_value(
                        values[f"reference_pct__{feature}"]
                    ),
                    "match_percentile": _python_value(
                        values[f"candidate_pct__{feature}"]
                    ),
                    "percentile_gap": _python_value(values[f"difference__{feature}"]),
                    "distance_contribution": _python_value(
                        values[f"contribution__{feature}"]
                    ),
                }
            )
        matches.append(
            {
                "rank": match_index,
                "player_id": int(values["player_id"]),
                "player_name": str(values["player_name"]),
                "team": str(values["team"]),
                "position": str(values["position"]),
                "similarity_score": _python_value(values["similarity_score"]),
                "weighted_distance": _python_value(values["distance"]),
                "feature_evidence": evidence,
            }
        )

    has_shot_features = any(
        item["feature"] == "rim_frequency"
        for match in matches
        for item in match["feature_evidence"]
    )
    limitations = [
        "A close statistical profile does not prove an identical basketball role.",
        "The expected-FG difficulty feature is a descriptive zone/action model, not a direct measurement of defender pressure."
        if has_shot_features
        else "Shot location, action type, and shot difficulty are not available in this dataset.",
        "Tracking, lineup, defensive assignment, and film context are not included.",
    ]
    return {
        "task": "explain deterministic player-neighbor retrieval",
        "data_status": data_status or "Bundled offline demo snapshot.",
        "reference_player": {
            "player_id": int(reference_profile["player_id"]),
            "player_name": str(reference_profile["player_name"]),
            "team": str(reference_profile["team"]),
            "position": str(reference_profile["position"]),
        },
        "retrieval_method": {
            "embedding": "league-percentile vector over structured season statistics",
            "distance": "role-weighted Euclidean distance",
            "preset": preset,
            "position_aware": bool(position_aware),
            "minimum_games": int(minimum_games),
            "minimum_shots": int(minimum_shots),
            "explanation_role": "rule-based summary only; retrieval fixes the ranking",
        },
        "matches": matches,
        "limitations": limitations,
    }


def evidence_frame(packet: dict) -> pd.DataFrame:
    """Return UI-ready evidence rows from one player packet."""

    frame = pd.DataFrame(packet.get("evidence", []))
    if frame.empty:
        return frame
    frame["display_value"] = frame.apply(
        lambda row: f"{row['value']:.1%}"
        if row["format"] == "percentage"
        else str(int(row["value"]))
        if row["format"] == "integer"
        else f"{row['value']:.1f}",
        axis=1,
    )
    return frame[["evidence_id", "label", "display_value", "context"]]


def build_team_needs_evidence(
    analysis: dict,
    candidates: pd.DataFrame,
    rotation_candidates: pd.DataFrame,
    model_evaluation: dict | None = None,
) -> dict:
    """Serialize fixed team-gap and player-fit results for explanation only."""

    team = analysis["team"]
    commonalities = []
    for index, row in enumerate(analysis["commonalities"].head(5).itertuples(), start=1):
        commonalities.append(
            {
                "evidence_id": f"C{index}",
                "feature": row.feature,
                "label": row.label,
                "desired_direction": row.desired_direction,
                "elite_stability": _python_value(row.elite_stability),
                "model_importance": _python_value(row.importance),
            }
        )
    gaps = []
    for index, row in enumerate(analysis["gaps"].head(6).itertuples(), start=1):
        gaps.append(
            {
                "evidence_id": f"G{index}",
                "feature": row.feature,
                "label": row.label,
                "team_value": _python_value(row.team_value),
                "elite_median": _python_value(row.elite_median),
                "desired_direction": row.desired_direction,
                "standardized_gap": _python_value(row.gap_z),
                "priority": _python_value(row.priority),
                "needed_player_trait": row.player_trait_label,
            }
        )

    def candidate_rows(frame: pd.DataFrame, prefix: str) -> list[dict]:
        rows = []
        for index, row in enumerate(frame.head(5).itertuples(), start=1):
            rows.append(
                {
                    "evidence_id": f"{prefix}{index}",
                    "player_id": int(row.player_id),
                    "player_name": str(row.player_name),
                    "team": str(row.team),
                    "position": str(row.position),
                    "fit_score": _python_value(row.fit_score),
                    "selection_rate": _python_value(
                        getattr(row, "selection_rate", 1.0)
                    ),
                    "games_played": int(row.games_played),
                    "minutes_per_game": _python_value(row.minutes_per_game),
                    "fit_reasons": str(row.fit_reasons),
                }
            )
        return rows

    return {
        "task": "explain a deterministic team-needs and player-fit analysis",
        "team": {key: _python_value(value) for key, value in team.items()},
        "benchmark": {
            "top_n": int(analysis["top_n"]),
            "seasons": list(analysis["benchmark_seasons"]),
            "ranking_basis": "regular-season win percentage",
            "model": (model_evaluation or {}).get(
                "selected_model", "Explainable logistic regression"
            ),
        },
        "elite_commonalities": commonalities,
        "team_gaps": gaps,
        "candidate_examples": candidate_rows(candidates, "P"),
        "rotation_candidate_examples": candidate_rows(rotation_candidates, "R"),
        "model_evaluation": model_evaluation or {},
        "limitations": [
            "The classifier recognizes elite statistical profiles; it does not prove which traits caused wins.",
            "Candidate fit is a transparent trait-alignment score, not a trade, contract, availability, or chemistry model.",
            "Individual defense uses box-score activity and rebounding proxies because player tracking and assignments are unavailable.",
            "The rule-based summary reports fixed gaps and rankings without altering them.",
        ],
    }


def build_roster_simulation_evidence(
    *,
    season: str,
    baseline_team: dict,
    baseline_prediction: dict,
    simulated_prediction: dict,
    roster: pd.DataFrame,
    added_players: list[str],
    removed_players: list[str],
    changes: pd.DataFrame,
    profiles: pd.DataFrame,
    neighbours: pd.DataFrame,
    model_evaluation: dict,
    support: dict | None = None,
    talent: dict | None = None,
    ceiling: dict | None = None,
    forced_players: list[str] | None = None,
) -> dict:
    """Serialize a fixed roster counterfactual for explanation only."""

    support = support or {"is_supported": True, "status": "not_checked"}
    talent = talent or {}
    ceiling = ceiling or {}
    is_counterfactual = bool(added_players or removed_players)
    quality = [
        {
            "evidence_id": "Q1",
            "label": "Baseline reference Net Rating",
            "value": _python_value(baseline_prediction["net_rating"]),
        },
        {
            "evidence_id": "Q2",
            "label": "Simulated Net Rating estimate",
            "value": _python_value(simulated_prediction["net_rating"]),
        },
        {
            "evidence_id": "Q3",
            "label": "Model-estimated Net Rating change",
            "value": _python_value(
                simulated_prediction["net_rating"] - baseline_prediction["net_rating"]
            ),
        },
        {
            "evidence_id": "Q4",
            "label": "Simulated 80% empirical win range",
            "low": _python_value(simulated_prediction["wins_low"]),
            "high": _python_value(simulated_prediction["wins_high"]),
        },
    ] if support.get("is_supported", True) and not is_counterfactual else []
    change_rows = [
        {
            "evidence_id": f"C{index}",
            "trait": str(row.trait),
            "baseline_percentile": _python_value(row.baseline),
            "simulated_percentile": _python_value(row.simulated),
            "change": _python_value(row.change),
        }
        for index, row in enumerate(changes.head(6).itertuples(), start=1)
    ]
    profile_rows = [
        {
            "evidence_id": f"P{index}",
            "profile": str(row.profile),
            "score": _python_value(row.score),
        }
        for index, row in enumerate(profiles.head(5).itertuples(), start=1)
    ]
    neighbour_rows = [
        {
            "evidence_id": f"N{index}",
            "team": str(row.team),
            "season": str(row.season),
            "wins": int(row.wins),
            "games": int(row.games),
            "net_rating": _python_value(row.net_rating),
            "embedding_similarity": _python_value(row.similarity),
        }
        for index, row in enumerate(neighbours.head(5).itertuples(), start=1)
    ]
    rotation = [
        {
            "evidence_id": f"R{index}",
            "player_id": int(row.player_id),
            "player_name": str(row.player_name),
            "source_team": str(row.team),
            "games_played": int(row.games_played),
            "observed_minutes_per_game": _python_value(row.minutes_per_game),
            "automatic_rotation_minutes": _python_value(
                getattr(row, "normalized_minutes", 240 * row.model_rotation_share)
            ),
            "model_rotation_share": _python_value(row.model_rotation_share),
            "modeled_rotation_impact": _python_value(
                getattr(row, "rotation_value", None)
            ),
            "role_context_percentile": _python_value(row.role_strength),
            "data_reliability": _python_value(row.data_reliability),
        }
        for index, row in enumerate(roster.itertuples(), start=1)
    ]
    delta = float(simulated_prediction["net_rating"] - baseline_prediction["net_rating"])
    resolution = float(
        model_evaluation.get(
            "minimum_detectable_change", model_evaluation.get("selected_model_mae", 0)
        )
        or 0
    )
    return {
        "task": "explain a fixed roster-embedding counterfactual",
        "season": season,
        "baseline_team": {key: _python_value(value) for key, value in baseline_team.items()},
        "reference_result_context": {
            "evidence_id": "W1",
            "actual_82_game_win_pace": _python_value(
                baseline_team.get("actual_82_game_win_pace")
            ),
            "net_rating_implied_wins": _python_value(
                baseline_team.get("net_rating_implied_wins")
            ),
            "actual_minus_implied_wins": _python_value(
                baseline_team.get("actual_minus_implied_wins")
            ),
            "interpretation": (
                "retrospective comparison of actual win pace with the wins historically "
                "associated with the team's actual Net Rating; not a preseason forecast"
            ),
        },
        "roster_moves": {
            "added": list(added_players),
            "removed": list(removed_players),
            "forced_into_rotation": list(forced_players or []),
        },
        "quality_estimates": quality,
        "counterfactual_validation": {
            "evidence_id": "V1",
            "is_counterfactual": is_counterfactual,
            "outcome_validated": False if is_counterfactual else None,
            "interpretation": (
                "roster identity and monotonic talent upside are available, but historical transactions have not validated an expected Net Rating or win change"
                if is_counterfactual
                else "unchanged observed-team baseline"
            ),
        },
        "model_support": {
            "evidence_id": "S1",
            "status": support.get("status", "not_checked"),
            "is_supported": bool(support.get("is_supported", True)),
            "nearest_historical_distance": support.get("nearest_distance"),
            "support_boundary": support.get("support_threshold"),
            "distance_to_boundary_ratio": support.get("distance_ratio"),
            "features_outside_training_range": support.get(
                "features_outside_training_range", []
            ),
            "interpretation": (
                "quality and win estimates may be displayed"
                if support.get("is_supported", True)
                else "quality and win estimates are withheld because this is unsupported extrapolation"
            ),
        },
        "roster_talent": {
            "evidence_id": "T1",
            "talent_index": talent.get("talent_index"),
            "historical_percentile": talent.get("historical_percentile"),
            "interpretation": "descriptive roster-talent comparison, not a win forecast",
        },
        "sandbox_upside": {
            "evidence_id": "U1",
            "talent_ceiling_net_rating": ceiling.get("net_rating"),
            "talent_ceiling_wins": ceiling.get("display_wins"),
            "experimental_expected_wins": ceiling.get("display_wins"),
            "talent_ceiling_losses": (
                82 - int(ceiling["display_wins"])
                if ceiling.get("display_wins") is not None
                else None
            ),
            "interpretation": (
                "optimistic monotonic talent-only point estimate for roster comparison; "
                "not yet calibrated on historical transaction outcomes"
            ),
        },
        "largest_embedding_changes": change_rows,
        "simulated_team_profiles": profile_rows,
        "nearest_historical_teams": neighbour_rows,
        "normalized_rotation": rotation,
        "model": {
            "selected_model": model_evaluation.get("selected_model", "unknown"),
            "validation_mae": model_evaluation.get("selected_model_mae"),
            "validation_method": "leave-one-season-out",
            "training_rows": model_evaluation.get("training_rows"),
            "seasons": model_evaluation.get("seasons", []),
            "explanation_role": "rule-based summary only; it cannot change the simulation",
        },
        "change_detection": {
            "raw_net_rating_change": (
                delta
                if support.get("is_supported", True) and not is_counterfactual
                else None
            ),
            "resolution_threshold": resolution,
            "interpretation": (
                "counterfactual outcome is not historically validated; do not infer a trade effect"
                if is_counterfactual
                else "unsupported extrapolation; do not interpret the raw regression output"
                if not support.get("is_supported", True)
                else (
                    "detectable directional change"
                    if abs(delta) >= resolution
                    else "no detectable change at current validation accuracy"
                )
            ),
        },
        "limitations": model_evaluation.get("limitations", [])
        + [
            "Win output is an uncertainty range derived from a Net Rating association, not a promise.",
            "Source-team MPG is an upper bound for additions; automatic minutes are earned only against lower-impact players at overlapping positions.",
            "The sample-shrunk plus/minus input is team context, not a causal individual impact estimate.",
            "Contracts, injuries, availability, coaching, chemistry, and transaction rules are not modeled.",
            "For rosters beyond the empirical training boundary, identity remains descriptive but quality forecasts are withheld.",
        ],
    }
