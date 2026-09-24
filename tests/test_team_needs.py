"""Tests for team fingerprints, gap analysis, and player-fit retrieval."""

from __future__ import annotations

import pandas as pd

from ball_ai.ai.grounding import build_team_needs_evidence
from ball_ai.ai.report_generator import generate_team_needs_report
from ball_ai.analytics.team_needs import (
    TEAM_FEATURES,
    analyze_team_needs,
    build_team_season_features,
    recommend_consensus_players,
    recommend_players,
)


def _team_game(
    game_id: int,
    team_id: int,
    opponent_id: int,
    team: str,
    score: int,
    opponent_score: int,
) -> dict:
    return {
        "season": "2024-25", "game_id": str(game_id), "game_date": "2025-01-01",
        "team_id": team_id, "opponent_team_id": opponent_id, "team": team,
        "team_score": score, "opponent_score": opponent_score,
        "field_goals_attempted": 80, "field_goals_made": 40,
        "three_pointers_attempted": 30, "three_pointers_made": 12,
        "free_throws_attempted": 20, "free_throws_made": 16,
        "rebounds_offensive": 10, "rebounds_defensive": 30, "turnovers": 12,
        "assists": 25, "steals": 8, "blocks": 5, "points_in_the_paint": 46,
        "points_fast_break": 15, "points_second_chance": 13, "bench_points": 34,
    }


def test_team_features_pair_opponents_and_calculate_ratings() -> None:
    games = pd.DataFrame(
        [
            _team_game(1, 1610612738, 1610612747, "Boston Celtics", 110, 100),
            _team_game(1, 1610612747, 1610612738, "Los Angeles Lakers", 100, 110),
        ]
    )

    features = build_team_season_features(games)
    boston = features.loc[features["team"].eq("Boston Celtics")].iloc[0]

    assert boston["wins"] == 1
    assert boston["effective_field_goal_percentage"] == (40 + 0.5 * 12) / 80
    assert boston["offensive_rating"] > boston["defensive_rating"]


def _synthetic_team_features() -> pd.DataFrame:
    rows = []
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
    for season in seasons:
        for rank in range(1, 13):
            strength = (13 - rank) / 12
            row = {
                "season": season, "team_id": rank, "team": f"Team {rank}",
                "team_abbreviation": f"T{rank}", "games": 82, "wins": 60-rank,
                "win_percentage": (60-rank)/82, "league_rank": rank,
                "offensive_rating": 105 + 10 * strength,
                "defensive_rating": 115 - 8 * strength,
                "net_rating": -10 + 18 * strength,
            }
            for index, feature in enumerate(TEAM_FEATURES):
                lower_is_good = feature in {
                    "opponent_effective_field_goal_percentage", "turnover_percentage"
                }
                row[feature] = (1-strength if lower_is_good else strength) + index * .001
            rows.append(row)
    return pd.DataFrame(rows)


def _candidate_players() -> pd.DataFrame:
    rows = []
    for player_id, team, multiplier in ((1, "AAA", 1.0), (2, "BBB", .6), (3, "T12", .9)):
        rows.append(
            {
                "player_id": player_id, "player_name": f"Player {player_id}",
                "team": team, "position": "G", "games_played": 60,
                "minutes_per_game": 25, "points_per_game": 15 * multiplier,
                "rebounds_per_game": 6 * multiplier, "assists_per_game": 5 * multiplier,
                "steals_per_game": 1.5 * multiplier, "blocks_per_game": multiplier,
                "turnovers_per_game": 2, "effective_field_goal_percentage": .55 * multiplier,
                "three_point_percentage": .38 * multiplier,
                "three_point_attempts_per_game": 6 * multiplier,
                "offensive_rebounds_per_game": 2 * multiplier,
                "defensive_rebounds_per_game": 5 * multiplier,
                "free_throw_attempts_per_game": 5 * multiplier,
                "two_point_percentage": .58 * multiplier,
                "rim_frequency": .3 * multiplier, "paint_frequency": .1 * multiplier,
                "three_point_frequency": .5 * multiplier, "dunk_frequency": .05 * multiplier,
                "layup_frequency": .25 * multiplier, "floater_frequency": .05 * multiplier,
            }
        )
    return pd.DataFrame(rows)


def test_gap_model_and_retrieval_are_fixed_before_generation() -> None:
    analysis = analyze_team_needs(
        _synthetic_team_features(), season="2024-25", team_id=12, top_n=5, lookback_seasons=5
    )
    candidates = recommend_players(_candidate_players(), analysis, top_k=3)

    assert analysis["gaps"].iloc[0]["priority"] > 0
    assert not candidates.empty
    assert "T12" not in candidates["team"].tolist()
    assert candidates.iloc[0]["player_name"] == "Player 1"

    packet = build_team_needs_evidence(analysis, candidates, candidates)
    report = generate_team_needs_report(packet, prefer_llm=False)
    assert report.mode == "template"
    assert "## Priority gaps" in report.text
    assert "[G1]" in report.text
    assert "[P1]" in report.text


def test_consensus_recommendations_expose_stability_and_position_diversity() -> None:
    candidates = recommend_consensus_players(
        _candidate_players(),
        _synthetic_team_features(),
        season="2024-25",
        team_id=12,
        top_k=3,
    )

    assert candidates["selection_rate"].between(0, 1).all()
    assert candidates["robust_fit_score"].le(candidates["fit_score"]).all()
