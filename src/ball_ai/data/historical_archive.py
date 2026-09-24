"""Build a local, season-partitioned Parquet archive of historical NBA data.

The archive intentionally separates data availability from data values. Basic
box scores extend to the league's early seasons, while shot location and action
labels only exist from 1996-97 onward. Missing shot fields are therefore absent,
not silently imputed as zero.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ball_ai.analytics.shot_profile import build_shot_profiles
from ball_ai.data.kaggle_provider import TEAM_ABBREVIATIONS
from ball_ai.data.shot_archive_provider import SHOT_COLUMNS


BOX_COLUMN_ALIASES = {
    "personId": "player_id",
    "gameId": "game_id",
    "gameDate": "game_date",
    "gameType": "game_type",
    "numMinutes": "minutes",
    "points": "points",
    "assists": "assists",
    "blocks": "blocks",
    "steals": "steals",
    "turnovers": "turnovers",
    "reboundsOffensive": "offensive_rebounds",
    "reboundsDefensive": "defensive_rebounds",
    "reboundsTotal": "rebounds",
    "foulsPersonal": "personal_fouls",
    "fieldGoalsAttempted": "field_goal_attempts",
    "fieldGoalsMade": "field_goals_made",
    "threePointersAttempted": "three_point_attempts",
    "threePointersMade": "three_points_made",
    "freeThrowsAttempted": "free_throw_attempts",
    "freeThrowsMade": "free_throws_made",
    "plusMinusPoints": "plus_minus",
    "playerteamId": "team_id",
    "opponentteamId": "opponent_id",
    "firstName": "first_name",
    "lastName": "last_name",
}

CORE_SOURCE_COLUMNS = {
    "personId", "gameId", "gameDate", "gameType", "numMinutes", "points",
    "assists", "blocks", "steals", "turnovers", "reboundsTotal",
    "fieldGoalsAttempted", "fieldGoalsMade", "threePointersAttempted",
    "threePointersMade", "freeThrowsAttempted", "freeThrowsMade",
    "playerteamId", "opponentteamId", "firstName", "lastName",
}

OPTIONAL_SOURCE_COLUMNS = {
    "reboundsOffensive", "reboundsDefensive", "foulsPersonal", "plusMinusPoints",
}

COUNTING_STATS = [
    "field_goals_made", "field_goal_attempts", "three_points_made",
    "three_point_attempts", "free_throws_made", "free_throw_attempts",
    "offensive_rebounds", "defensive_rebounds", "rebounds", "assists", "steals",
    "blocks", "turnovers", "personal_fouls", "points", "plus_minus",
]


class HistoricalArchiveError(RuntimeError):
    """Raised when source data cannot be converted without losing integrity."""


def season_label_from_dates(dates: pd.Series) -> pd.Series:
    """Map game dates to NBA season labels such as ``2024-25``.

    Games from September through December belong to the season beginning that
    calendar year; January through August belong to the previous start year.
    """

    parsed = pd.to_datetime(dates, errors="coerce")
    starts = parsed.dt.year.where(parsed.dt.month >= 9, parsed.dt.year - 1)
    labels = starts.map(
        lambda year: pd.NA if pd.isna(year) else f"{int(year):04d}-{(int(year) + 1) % 100:02d}"
    )
    return labels.astype("string")


def normalize_season_type(values: pd.Series) -> pd.Series:
    """Normalize the source's game labels without discarding unusual events."""

    lowered = values.fillna("unknown").astype(str).str.strip().str.lower()
    result = pd.Series("other", index=values.index, dtype="string")
    result.loc[lowered.str.contains("regular", regex=False)] = "regular"
    result.loc[
        lowered.str.contains("in-season tournament", regex=False)
        | lowered.str.contains("in season tournament", regex=False)
    ] = "regular"
    result.loc[lowered.str.contains("playoff", regex=False)] = "playoffs"
    result.loc[lowered.str.contains("all.star", regex=True)] = "all_star"
    result.loc[lowered.eq("unknown")] = "unknown"
    return result


def _available_box_columns(csv_path: Path) -> list[str]:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = CORE_SOURCE_COLUMNS - set(header)
    if missing:
        raise HistoricalArchiveError(
            f"PlayerStatistics.csv is missing required columns: {sorted(missing)}"
        )
    wanted = set(CORE_SOURCE_COLUMNS) | OPTIONAL_SOURCE_COLUMNS
    return [column for column in header if column in wanted]


def normalize_box_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Normalize one source CSV chunk while preserving null historical fields."""

    frame = chunk.rename(columns=BOX_COLUMN_ALIASES).copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["season"] = season_label_from_dates(frame["game_date"])
    frame["source_game_type"] = frame["game_type"].astype("string")
    frame["season_type"] = normalize_season_type(frame.pop("game_type"))
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce").astype("Int64")
    frame["game_id"] = frame["game_id"].astype("string")
    frame["player_name"] = (
        frame.pop("first_name").fillna("").astype(str).str.strip()
        + " "
        + frame.pop("last_name").fillna("").astype(str).str.strip()
    ).str.strip()

    for source, target in (("team_id", "team"), ("opponent_id", "opponent")):
        ids = pd.to_numeric(frame.pop(source), errors="coerce").astype("Int64")
        frame[target] = ids.map(TEAM_ABBREVIATIONS).fillna("N/A")

    numeric = ["minutes", *COUNTING_STATS]
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["player_id", "game_date", "season"])
    # Early box scores can contain valid counting stats without recorded minutes,
    # while modern DNP rows often contain null minutes and an all-zero stat line.
    available_stats = [column for column in COUNTING_STATS if column in frame]
    has_recorded_activity = frame[available_stats].fillna(0).abs().sum(axis=1).gt(0)
    frame = frame.loc[frame["minutes"].gt(0) | has_recorded_activity].copy()
    frame["player_id"] = frame["player_id"].astype("int64")
    frame["game_date"] = frame["game_date"].dt.date.astype(str)
    leading = [
        "season", "season_type", "game_date", "game_id", "player_id",
        "player_name", "team", "opponent", "source_game_type", "minutes",
    ]
    return frame[[*leading, *[column for column in COUNTING_STATS if column in frame]]]


def _write_partition_parts(
    frame: pd.DataFrame, root: Path, chunk_number: int
) -> list[dict[str, object]]:
    coverage: list[dict[str, object]] = []
    for (season, season_type), group in frame.groupby(["season", "season_type"], sort=False):
        directory = root / f"season={season}" / f"season_type={season_type}"
        directory.mkdir(parents=True, exist_ok=True)
        payload = group.drop(columns=["season", "season_type"])
        path = directory / f"part-{chunk_number:04d}.parquet"
        payload.to_parquet(path, index=False, compression="snappy")
        coverage.append(
            {"season": season, "season_type": season_type, "box_score_rows": len(group)}
        )
    return coverage


def _snake_case(name: str) -> str:
    """Convert source camelCase labels to stable Parquet field names."""

    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()
    return value.replace("date_time_est", "datetime_est")


def build_team_box_score_archive(
    statistics_path: Path,
    destination: Path,
    *,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Preserve team box scores, including seasons before player rows exist."""

    header = pd.read_csv(statistics_path, nrows=0).columns.tolist()
    required = {"gameId", "gameDate", "gameType", "teamId", "teamCity", "teamName"}
    if missing := required - set(header):
        raise HistoricalArchiveError(
            f"TeamStatistics.csv is missing required columns: {sorted(missing)}"
        )
    root = destination / "team_game_stats"
    if root.exists():
        shutil.rmtree(root)
    coverage_rows: list[dict[str, object]] = []
    for number, chunk in enumerate(
        pd.read_csv(statistics_path, chunksize=chunk_size, low_memory=False)
    ):
        chunk = chunk.rename(columns={column: _snake_case(column) for column in chunk})
        chunk["game_date"] = pd.to_datetime(chunk["game_date"], errors="coerce")
        chunk["season"] = season_label_from_dates(chunk["game_date"])
        chunk["season_type"] = normalize_season_type(chunk.pop("game_type"))
        chunk["team"] = (
            chunk["team_city"].fillna("").astype(str).str.strip()
            + " "
            + chunk["team_name"].fillna("").astype(str).str.strip()
        ).str.strip()
        if {"opponent_team_city", "opponent_team_name"}.issubset(chunk.columns):
            chunk["opponent"] = (
                chunk["opponent_team_city"].fillna("").astype(str).str.strip()
                + " "
                + chunk["opponent_team_name"].fillna("").astype(str).str.strip()
            ).str.strip()
        chunk = chunk.dropna(subset=["game_date", "season", "team_id"])
        chunk["game_date"] = chunk["game_date"].dt.date.astype(str)
        coverage_rows.extend(_write_partition_parts(chunk, root, number))
    coverage = (
        pd.DataFrame(coverage_rows)
        .rename(columns={"box_score_rows": "team_box_score_rows"})
        .groupby(["season", "season_type"], as_index=False)["team_box_score_rows"]
        .sum()
    )
    return coverage


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.Series(
        np.divide(
            numerator.to_numpy(dtype=float),
            denominator.to_numpy(dtype=float),
            out=np.full(len(numerator), np.nan),
            where=denominator.to_numpy(dtype=float) != 0,
        ),
        index=numerator.index,
    )


def _partial_regular_season_aggregate(games: pd.DataFrame) -> pd.DataFrame:
    """Reduce a game chunk to a small mergeable player-season table."""

    regular = games.loc[games["season_type"].eq("regular")].copy()
    if regular.empty:
        return pd.DataFrame()
    regular["minutes_observed"] = regular["minutes"].notna().astype(int)
    regular["games_played"] = 1
    aggregations: dict[str, tuple[str, object]] = {
        "games_played": ("games_played", "sum"),
        "minutes_total": ("minutes", "sum"),
        "minutes_observed": ("minutes_observed", "sum"),
    }
    for column in COUNTING_STATS:
        if column in regular:
            aggregations[f"{column}_total"] = (
                column,
                lambda values: values.sum(min_count=1),
            )
    return (
        regular.groupby(["player_id", "player_name", "season"], as_index=False)
        .agg(**aggregations)
    )


def _aggregate_regular_seasons(parts: Iterable[pd.DataFrame]) -> pd.DataFrame:
    partials = [part for part in parts if not part.empty]
    if not partials:
        return pd.DataFrame()

    combined = pd.concat(partials, ignore_index=True)
    total_columns = [column for column in combined if column.endswith("_total")]
    totals = (
        combined.groupby(["player_id", "player_name", "season"], as_index=False)[
            ["games_played", "minutes_observed", *total_columns]
        ]
        .sum(min_count=1)
    )
    totals["minutes_per_game"] = _safe_divide(
        totals.pop("minutes_total"), totals.pop("minutes_observed")
    )
    attempts = {
        "field_goal_percentage": ("field_goals_made_total", "field_goal_attempts_total"),
        "three_point_percentage": ("three_points_made_total", "three_point_attempts_total"),
        "free_throw_percentage": ("free_throws_made_total", "free_throw_attempts_total"),
    }
    for output, (made, attempted) in attempts.items():
        totals[output] = _safe_divide(totals[made], totals[attempted])
    totals["two_points_made_total"] = (
        totals["field_goals_made_total"] - totals["three_points_made_total"]
    )
    totals["two_point_attempts_total"] = (
        totals["field_goal_attempts_total"] - totals["three_point_attempts_total"]
    )
    totals["two_point_percentage"] = _safe_divide(
        totals["two_points_made_total"], totals["two_point_attempts_total"]
    )
    totals["effective_field_goal_percentage"] = _safe_divide(
        totals["field_goals_made_total"] + 0.5 * totals["three_points_made_total"],
        totals["field_goal_attempts_total"],
    )
    for column in COUNTING_STATS:
        total = f"{column}_total"
        if total in totals:
            totals[f"{column}_per_game"] = _safe_divide(
                totals[total], totals["games_played"]
            )
    return totals.sort_values(["player_name", "season"]).reset_index(drop=True)


def build_box_score_archive(
    statistics_path: Path,
    destination: Path,
    *,
    chunk_size: int = 200_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the historical CSV to partitioned Parquet and season aggregates."""

    columns = _available_box_columns(statistics_path)
    games_root = destination / "player_game_stats"
    if games_root.exists():
        shutil.rmtree(games_root)
    coverage_rows: list[dict[str, object]] = []
    partial_aggregates: list[pd.DataFrame] = []
    for number, chunk in enumerate(
        pd.read_csv(statistics_path, usecols=columns, chunksize=chunk_size, low_memory=False)
    ):
        normalized = normalize_box_chunk(chunk)
        coverage_rows.extend(_write_partition_parts(normalized, games_root, number))
        partial_aggregates.append(_partial_regular_season_aggregate(normalized))

    seasons = _aggregate_regular_seasons(partial_aggregates)
    if seasons.empty:
        raise HistoricalArchiveError("No regular-season box scores were produced.")
    seasons.to_parquet(
        destination / "player_season_stats.parquet", index=False, compression="snappy"
    )
    coverage = (
        pd.DataFrame(coverage_rows)
        .groupby(["season", "season_type"], as_index=False)["box_score_rows"]
        .sum()
    )
    return seasons, coverage


def rebuild_box_score_aggregates(
    destination: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove all-zero DNP rows and rebuild aggregates from stored Parquet parts."""

    games_root = destination / "player_game_stats"
    files = sorted(games_root.rglob("*.parquet"))
    if not files:
        raise HistoricalArchiveError(f"No player game Parquet files found in {games_root}.")
    partials: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    for path in files:
        games = pd.read_parquet(path)
        available_stats = [column for column in COUNTING_STATS if column in games]
        activity = games[available_stats].fillna(0).abs().sum(axis=1).gt(0)
        games = games.loc[games["minutes"].gt(0) | activity].copy()
        temporary = path.with_suffix(".parquet.tmp")
        games.to_parquet(temporary, index=False, compression="snappy")
        temporary.replace(path)

        season = next(part.split("=", 1)[1] for part in path.parts if part.startswith("season="))
        season_type = next(
            part.split("=", 1)[1]
            for part in path.parts
            if part.startswith("season_type=")
        )
        enriched = games.assign(season=season, season_type=season_type)
        if "source_game_type" in enriched:
            enriched["season_type"] = normalize_season_type(
                enriched["source_game_type"]
            )
        elif season == "2023-24" and season_type == "other":
            # Compatibility repair for archives created before source_game_type
            # was retained: 2023 tournament games through Dec 8 count in the
            # regular-season standings; preseason and the Dec 9 final do not.
            dates = pd.to_datetime(enriched["game_date"], errors="coerce")
            tournament = (
                dates.between("2023-11-03", "2023-12-08")
                & enriched["opponent"].ne("N/A")
            )
            enriched.loc[tournament, "season_type"] = "regular"
        partials.append(_partial_regular_season_aggregate(enriched))
        coverage_rows.append(
            {
                "season": season,
                "season_type": season_type,
                "box_score_rows": len(games),
            }
        )

    seasons = _aggregate_regular_seasons(partials)
    seasons.to_parquet(
        destination / "player_season_stats.parquet", index=False, compression="snappy"
    )
    coverage = (
        pd.DataFrame(coverage_rows)
        .groupby(["season", "season_type"], as_index=False)["box_score_rows"]
        .sum()
    )
    return seasons, coverage


def build_players_archive(players_path: Path, destination: Path) -> pd.DataFrame:
    """Store source player biographies as Parquet without narrowing their schema."""

    players = pd.read_csv(players_path, low_memory=False)
    if "personId" not in players:
        raise HistoricalArchiveError("Players.csv is missing personId.")
    players = players.rename(columns={"personId": "player_id"})
    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce").astype("Int64")
    players = players.dropna(subset=["player_id"]).drop_duplicates("player_id")
    players["player_id"] = players["player_id"].astype("int64")
    players.to_parquet(destination / "players.parquet", index=False, compression="snappy")
    return players


def build_shot_archive(
    raw_shots_path: Path, destination: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve all shot events and create one style profile per player-season."""

    raw_destination = destination / "shot_events.parquet"
    if raw_shots_path.resolve() != raw_destination.resolve():
        shutil.copy2(raw_shots_path, raw_destination)

    schema = pq.ParquetFile(raw_destination).schema_arrow
    required = {*SHOT_COLUMNS, "_season", "_season_type"}
    if missing := required - set(schema.names):
        raise HistoricalArchiveError(
            f"Shot archive is missing required columns: {sorted(missing)}"
        )

    season_keys = pd.read_parquet(
        raw_destination, columns=["_season", "_season_type"]
    )
    season_keys["season"] = season_keys["_season"].map(
        lambda year: f"{int(year):04d}-{(int(year) + 1) % 100:02d}"
    )
    season_keys["season_type"] = season_keys["_season_type"].map(
        {"rg": "regular", "po": "playoffs"}
    ).fillna(season_keys["_season_type"].astype(str))
    coverage = (
        season_keys.groupby(["season", "season_type"], as_index=False)
        .size()
        .rename(columns={"size": "shot_event_rows"})
    )

    profiles: list[pd.DataFrame] = []
    regular_years = sorted(
        season_keys.loc[season_keys["_season_type"].eq("rg"), "_season"].unique()
    )
    for year in regular_years:
        raw = pd.read_parquet(
            raw_destination,
            columns=SHOT_COLUMNS,
            filters=[[('_season', '=', int(year)), ('_season_type', '=', 'rg')]],
        )
        shots = raw.rename(
            columns={
                "PLAYER_ID": "player_id",
                "SHOT_MADE_FLAG": "shot_made",
                "SHOT_DISTANCE": "shot_distance",
                "ACTION_TYPE": "action_type",
                "SHOT_ZONE_BASIC": "shot_zone_basic",
            }
        )
        shots["player_id"] = pd.to_numeric(shots["player_id"], errors="coerce")
        shots = shots.dropna(subset=["player_id"])
        shots["player_id"] = shots["player_id"].astype(int)
        profile = build_shot_profiles(shots)
        profile.insert(1, "season", f"{int(year):04d}-{(int(year) + 1) % 100:02d}")
        profiles.append(profile)
    profile_frame = pd.concat(profiles, ignore_index=True)
    profile_frame.to_parquet(
        destination / "player_shot_profiles.parquet", index=False, compression="snappy"
    )
    return profile_frame, coverage


def write_archive_metadata(
    destination: Path,
    *,
    sources: dict[str, str],
    box_coverage: pd.DataFrame | None = None,
    team_coverage: pd.DataFrame | None = None,
    shot_coverage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Write queryable coverage plus human-readable provenance metadata."""

    existing_path = destination / "coverage.parquet"
    existing = pd.read_parquet(existing_path) if existing_path.exists() else None
    if box_coverage is None and existing is not None and "box_score_rows" in existing:
        box_coverage = existing[
            ["season", "season_type", "box_score_rows"]
        ].dropna(subset=["box_score_rows"])
    if shot_coverage is None and existing is not None and "shot_event_rows" in existing:
        shot_coverage = existing[
            ["season", "season_type", "shot_event_rows"]
        ].dropna(subset=["shot_event_rows"])
    if team_coverage is None and existing is not None and "team_box_score_rows" in existing:
        team_coverage = existing[
            ["season", "season_type", "team_box_score_rows"]
        ].dropna(subset=["team_box_score_rows"])

    frames = [
        frame for frame in (box_coverage, team_coverage, shot_coverage) if frame is not None
    ]
    if not frames:
        raise HistoricalArchiveError("No archive coverage was supplied or already stored.")
    coverage = frames[0]
    for frame in frames[1:]:
        coverage = coverage.merge(frame, on=["season", "season_type"], how="outer")
    coverage = coverage.sort_values(["season", "season_type"]).reset_index(drop=True)
    coverage.to_parquet(destination / "coverage.parquet", index=False, compression="snappy")
    metadata_path = destination / "metadata.json"
    previous_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "format": "Apache Parquet (Snappy compression)",
        "partitioning": "player_game_stats/season=<season>/season_type=<type>/",
        "missingness_policy": (
            "Unavailable historical fields remain null or absent; shot-detail coverage "
            "must never be inferred before 1996-97."
        ),
        "sources": {**previous_metadata.get("sources", {}), **sources},
        "row_counts": {
            column: int(coverage[column].fillna(0).sum())
            for column in ("box_score_rows", "team_box_score_rows", "shot_event_rows")
            if column in coverage
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return coverage
