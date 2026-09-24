"""Tests for team and name player discovery."""

from __future__ import annotations

import pandas as pd

from ball_ai.data.player_filters import (
    filter_players,
    order_player_candidates,
    resolve_player_search,
    strict_player_name_matches,
)


PLAYERS = pd.DataFrame(
    [
        {"player_id": 1, "player_name": "Jalen Williams", "team": "OKC", "position": "F"},
        {"player_id": 2, "player_name": "Jaylin Williams", "team": "OKC", "position": "C"},
        {"player_id": 3, "player_name": "Jalen Brunson", "team": "NYK", "position": "G"},
    ]
)


def test_filter_players_by_team() -> None:
    result = filter_players(PLAYERS, team="OKC")
    assert result["player_id"].tolist() == [1, 2]


def test_filter_players_by_name_is_case_insensitive_and_literal() -> None:
    result = filter_players(PLAYERS, query="JALEN")
    assert result["player_id"].tolist() == [1, 3]
    assert filter_players(PLAYERS, query="[").empty


def test_team_and_name_filters_can_be_combined() -> None:
    result = filter_players(PLAYERS, team="OKC", query="Jaylin")
    assert result["player_id"].tolist() == [2]


def test_search_recommends_other_teams_when_selected_team_has_no_match() -> None:
    result = resolve_player_search(PLAYERS, team="NYK", query="Jaylin")
    assert result.used_league_wide_fallback is True
    assert result.matches["player_id"].tolist() == [2]


def test_search_keeps_team_priority_when_local_match_exists() -> None:
    result = resolve_player_search(PLAYERS, team="OKC", query="Jalen")
    assert result.used_league_wide_fallback is False
    assert result.matches["player_id"].tolist() == [1]


def test_typeahead_candidates_restrict_team_and_exclude_queued_players() -> None:
    candidates = order_player_candidates(
        PLAYERS,
        preferred_team="OKC",
        excluded_ids=[2],
    )
    assert 2 not in candidates["player_id"].tolist()
    assert candidates.iloc[0]["team"] == "OKC"
    assert set(candidates["team"]) == {"OKC"}


def test_strict_name_matching_accepts_small_typos_without_irrelevant_names() -> None:
    players = pd.concat(
        [
            PLAYERS,
            pd.DataFrame(
                [
                    {"player_id": 4, "player_name": "Luka Doncic", "team": "LAL", "position": "G"},
                    {"player_id": 5, "player_name": "DeMar DeRozan", "team": "SAC", "position": "F"},
                ]
            ),
        ],
        ignore_index=True,
    )
    assert strict_player_name_matches(players, "doncic")["player_id"].tolist() == [4]
    assert strict_player_name_matches(players, "doni")["player_id"].tolist() == [4]


def test_exact_name_token_ranks_before_a_longer_prefix() -> None:
    players = pd.DataFrame(
        [
            {"player_id": 1, "player_name": "John Konchar", "team": "UTA", "position": "G"},
            {"player_id": 2, "player_name": "Kon Knueppel", "team": "CHA", "position": "G-F"},
        ]
    )
    assert resolve_player_search(players, query="kon").matches["player_id"].tolist() == [2, 1]
