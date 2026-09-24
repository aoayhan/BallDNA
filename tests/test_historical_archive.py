"""Tests for the tiered historical Parquet archive."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ball_ai.data.historical_archive import (
    build_box_score_archive,
    build_shot_archive,
    build_team_box_score_archive,
    normalize_season_type,
    season_label_from_dates,
    write_archive_metadata,
)


def _game(game_id: int, date: str, game_type: str = "Regular Season") -> dict:
    return {
        "firstName": "Historic", "lastName": "Player", "personId": 10,
        "gameId": game_id, "gameDate": date, "gameType": game_type,
        "numMinutes": 30, "points": 20, "assists": 5, "blocks": 1,
        "steals": 1, "turnovers": 2, "reboundsTotal": 7,
        "fieldGoalsAttempted": 15, "fieldGoalsMade": 8,
        "threePointersAttempted": 5, "threePointersMade": 2,
        "freeThrowsAttempted": 3, "freeThrowsMade": 2,
        "playerteamId": 1610612738, "opponentteamId": 1610612747,
    }


def test_season_labels_cross_calendar_year() -> None:
    labels = season_label_from_dates(pd.Series(["2024-10-01", "2025-04-01"]))
    assert labels.tolist() == ["2024-25", "2024-25"]


def test_in_season_tournament_counts_as_regular_season() -> None:
    result = normalize_season_type(
        pd.Series(["Regular Season", "In-Season Tournament", "Playoffs"])
    )
    assert result.tolist() == ["regular", "regular", "playoffs"]


def test_early_box_score_without_minutes_is_preserved(tmp_path: Path) -> None:
    early = _game(1, "1946-11-01")
    early["numMinutes"] = None
    source = tmp_path / "PlayerStatistics.csv"
    pd.DataFrame([early]).to_csv(source, index=False)

    seasons, _ = build_box_score_archive(source, tmp_path)

    assert seasons.iloc[0]["season"] == "1946-47"
    assert seasons.iloc[0]["games_played"] == 1
    assert pd.isna(seasons.iloc[0]["minutes_per_game"])


def test_modern_dnp_with_empty_stat_line_is_excluded(tmp_path: Path) -> None:
    dnp = _game(1, "2024-11-01")
    dnp["numMinutes"] = None
    for column in (
        "points", "assists", "blocks", "steals", "turnovers", "reboundsTotal",
        "fieldGoalsAttempted", "fieldGoalsMade", "threePointersAttempted",
        "threePointersMade", "freeThrowsAttempted", "freeThrowsMade",
    ):
        dnp[column] = 0
    played = _game(2, "2024-11-02")
    source = tmp_path / "PlayerStatistics.csv"
    pd.DataFrame([dnp, played]).to_csv(source, index=False)

    seasons, coverage = build_box_score_archive(source, tmp_path)

    assert seasons.iloc[0]["games_played"] == 1
    assert coverage["box_score_rows"].sum() == 1


def test_box_archive_is_partitioned_and_aggregated(tmp_path: Path) -> None:
    source = tmp_path / "PlayerStatistics.csv"
    pd.DataFrame([_game(1, "2024-10-01"), _game(2, "2025-04-01")]).to_csv(
        source, index=False
    )

    seasons, coverage = build_box_score_archive(source, tmp_path, chunk_size=1)

    assert (tmp_path / "player_game_stats/season=2024-25/season_type=regular").is_dir()
    assert seasons.iloc[0]["games_played"] == 2
    assert seasons.iloc[0]["points_per_game"] == 20
    assert coverage["box_score_rows"].sum() == 2


def test_shots_keep_raw_events_and_add_profiles(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            (10, 1, 2, "Driving Dunk Shot", "Restricted Area", 1996, "rg"),
            (10, 0, 25, "Step Back Jump shot", "Above the Break 3", 1996, "rg"),
        ],
        columns=[
            "PLAYER_ID", "SHOT_MADE_FLAG", "SHOT_DISTANCE", "ACTION_TYPE",
            "SHOT_ZONE_BASIC", "_season", "_season_type",
        ],
    )
    source = tmp_path / "source.parquet"
    destination = tmp_path / "archive"
    destination.mkdir()
    raw.to_parquet(source, index=False)

    profiles, coverage = build_shot_archive(source, destination)

    assert (destination / "shot_events.parquet").exists()
    assert profiles.iloc[0]["season"] == "1996-97"
    assert profiles.iloc[0]["shot_attempts"] == 2
    assert coverage.iloc[0]["shot_event_rows"] == 2


def test_coverage_distinguishes_basic_and_shot_data(tmp_path: Path) -> None:
    box = pd.DataFrame(
        [{"season": "1946-47", "season_type": "regular", "box_score_rows": 12}]
    )
    shots = pd.DataFrame(
        [{"season": "1996-97", "season_type": "regular", "shot_event_rows": 20}]
    )

    coverage = write_archive_metadata(
        tmp_path,
        box_coverage=box,
        shot_coverage=shots,
        sources={"test": "fixture"},
    )

    early = coverage.loc[coverage["season"].eq("1946-47")].iloc[0]
    assert early["box_score_rows"] == 12
    assert pd.isna(early["shot_event_rows"])

    updated = write_archive_metadata(
        tmp_path, box_coverage=None, shot_coverage=shots, sources={"shots": "fixture-2"}
    )
    assert updated.loc[updated["season"].eq("1946-47"), "box_score_rows"].iloc[0] == 12


def test_team_archive_reaches_early_seasons(tmp_path: Path) -> None:
    source = tmp_path / "TeamStatistics.csv"
    pd.DataFrame(
        [
            {
                "gameId": 1,
                "gameDate": "1946-11-01",
                "gameType": "Regular Season",
                "teamId": 20,
                "teamCity": "Early",
                "teamName": "Club",
                "opponentTeamCity": "Other",
                "opponentTeamName": "Club",
                "points": 70,
            }
        ]
    ).to_csv(source, index=False)

    coverage = build_team_box_score_archive(source, tmp_path)

    assert coverage.iloc[0]["season"] == "1946-47"
    assert coverage.iloc[0]["team_box_score_rows"] == 1
    stored = pd.read_parquet(
        next((tmp_path / "team_game_stats").rglob("*.parquet"))
    )
    assert stored.iloc[0]["team"] == "Early Club"
