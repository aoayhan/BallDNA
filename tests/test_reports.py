"""Tests for no-key fallback generation and lightweight evaluation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import ball_ai.ai.llm_client as llm_client
from ball_ai.ai.grounding import build_player_evidence, build_similarity_evidence
from ball_ai.ai.prompts import REPORT_SECTIONS
from ball_ai.ai.quality import evaluate_report
from ball_ai.ai.report_generator import (
    fallback_answer,
    fallback_comparison,
    fallback_scouting_report,
    generate_scouting_report,
    generate_similarity_explanation,
)
from ball_ai.analytics.similarity import find_similar_players
from ball_ai.analytics.trends import calculate_trends
from ball_ai.data.database import (
    get_player_profile,
    get_players,
    get_recent_games,
    get_season_stats,
)


def test_fallback_report_is_complete_and_cited(demo_database: Path) -> None:
    player_id = int(get_players(demo_database).iloc[0].player_id)
    profile = get_player_profile(player_id, demo_database)
    packet = build_player_evidence(profile, get_recent_games(player_id, 10, demo_database))
    result = generate_scouting_report(packet, prefer_llm=False)

    assert result.mode == "template"
    assert "[E1]" in result.text
    for section in REPORT_SECTIONS:
        assert f"## {section}" in result.text


def test_fallback_report_passes_quality_checks(demo_database: Path) -> None:
    player_id = int(get_players(demo_database).iloc[0].player_id)
    profile = get_player_profile(player_id, demo_database)
    packet = build_player_evidence(profile, get_recent_games(player_id, 10, demo_database))
    report = generate_scouting_report(packet, prefer_llm=False).text
    checks = evaluate_report(report, packet)
    assert all(check["passed"] for check in checks.values()), checks


def test_missing_api_key_automatically_uses_fallback(
    demo_database: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        llm_client,
        "settings",
        SimpleNamespace(openai_api_key=None, openai_model="unused"),
    )
    player_id = int(get_players(demo_database).iloc[0].player_id)
    packet = build_player_evidence(
        get_player_profile(player_id, demo_database),
        get_recent_games(player_id, 10, demo_database),
    )
    result = generate_scouting_report(packet, prefer_llm=True)
    assert result.mode == "template"
    assert result.warning == "OPENAI_API_KEY is not configured."


def test_evidence_packet_includes_prior_season_from_parquet_context(
    demo_database: Path,
) -> None:
    player_id = int(get_players(demo_database).iloc[0].player_id)
    profile = get_player_profile(player_id, demo_database)
    history = pd.DataFrame(
        [
            {
                "season": "2022-23",
                "points_per_game": 18,
                "rebounds_per_game": 6,
                "assists_per_game": 4,
                "three_point_percentage": .37,
            },
            {"season": profile["season"]},
        ]
    )

    packet = build_player_evidence(
        profile,
        get_recent_games(player_id, 10, demo_database),
        season_history=history,
    )

    assert packet["career_context"]["previous_season"] == "2022-23"
    assert any(
        item["metric"] == "previous_points_per_game"
        for item in packet["evidence"]
    )


def test_similarity_fallback_explains_fixed_matches(demo_database: Path) -> None:
    season = get_season_stats(demo_database)
    player_id = int(season.iloc[0].player_id)
    matches = find_similar_players(season, player_id, top_n=3)
    packet = build_similarity_evidence(
        get_player_profile(player_id, demo_database),
        matches,
        "Balanced profile",
        False,
        0,
    )

    result = generate_similarity_explanation(packet, prefer_llm=False)

    assert result.mode == "template"
    assert "## Why the matches are similar" in result.text
    assert "language layer did not choose or reorder" in result.text
    assert "[M1E" in result.text


def test_fallback_semantics_respect_role_and_sample_size() -> None:
    def packet(name: str, position: str, points: float, rebounds: float, assists: float,
               three: float, games: int, attempts: float, makes: float) -> dict:
        values = [
            ("points_per_game", points, "number"),
            ("rebounds_per_game", rebounds, "number"),
            ("assists_per_game", assists, "number"),
            ("turnovers_per_game", 2.0, "number"),
            ("three_point_percentage", three, "percentage"),
        ]
        return {
            "player": {"player_name": name, "position": position, "season": "2025-26"},
            "season_stats": {
                **{key: value for key, value, _ in values},
                "games_played": games,
                "three_point_attempts_per_game": attempts,
                "three_points_made_per_game": makes,
            },
            "evidence": [
                {"evidence_id": f"E{i}", "metric": key, "label": key,
                 "value": value, "format": kind, "context": "season"}
                for i, (key, value, kind) in enumerate(values, 1)
            ],
            "shot_profile": {}, "trends": [], "recent_summary": {"games": 4},
            "limitations": ["Test packet."],
        }

    watson = packet("Peyton Watson", "G", 10, 6, 1, 0, 50, 0, 0)
    watson_report = fallback_scouting_report(watson)
    assert "perimeter or secondary-guard role" in watson_report
    assert "Supported perimeter shooting" not in watson_report

    gobert = packet("Rudy Gobert", "C", 14, 12, 1, 0, 70, 0, 0)
    assert "finishing/rebounding center" in fallback_answer(
        gobert, "Is he a scorer or facilitator?"
    )

    mark = packet("Mark Williams", "C", 12, 8, 1, 1.0, 1, 1, 1)
    curry = packet("Stephen Curry", "G", 27, 4, 6, .40, 70, 10, 4)
    comparison = fallback_comparison({"players": [mark, curry]})
    assert "spacing-oriented need points to Stephen Curry" in comparison

    mark_evidence = build_player_evidence(
        {
            "player_id": 1, "player_name": "Mark Williams", "team": "CHA",
            "position": "C", "season": "2025-26", "games_played": 1,
            "points_per_game": 12, "rebounds_per_game": 8,
            "assists_per_game": 1, "turnovers_per_game": 1,
            "three_point_percentage": 1.0,
            "three_point_attempts_per_game": 1.0,
            "three_points_made_per_game": 1.0,
        },
        pd.DataFrame(),
    )
    assert "three_point_percentage" not in mark_evidence["season_stats"]

    four_games = pd.DataFrame(
        {
            "game_date": pd.date_range("2026-01-01", periods=4),
            "field_goals_made": [2] * 4, "three_points_made": [0] * 4,
            "field_goal_attempts": [5] * 4, "free_throw_attempts": [0] * 4,
            "points": [4] * 4, "rebounds": [2] * 4, "assists": [1] * 4,
            "turnovers": [1] * 4,
        }
    )
    assert calculate_trends(four_games) == []
