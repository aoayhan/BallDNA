"""SQLite persistence and read models for BallDNA."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from ball_ai.config import settings


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    player_name TEXT NOT NULL UNIQUE,
    team TEXT NOT NULL,
    position TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS player_game_stats (
    game_id TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    game_date TEXT NOT NULL,
    opponent TEXT NOT NULL,
    minutes REAL NOT NULL,
    points INTEGER NOT NULL,
    rebounds INTEGER NOT NULL,
    assists INTEGER NOT NULL,
    steals INTEGER NOT NULL,
    blocks INTEGER NOT NULL,
    turnovers INTEGER NOT NULL,
    field_goal_attempts INTEGER NOT NULL,
    field_goals_made INTEGER NOT NULL,
    three_point_attempts INTEGER NOT NULL,
    three_points_made INTEGER NOT NULL,
    free_throw_attempts INTEGER NOT NULL,
    free_throws_made INTEGER NOT NULL,
    PRIMARY KEY (game_id, player_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS player_season_stats (
    player_id INTEGER NOT NULL,
    season TEXT NOT NULL,
    games_played INTEGER NOT NULL,
    minutes_per_game REAL NOT NULL,
    points_per_game REAL NOT NULL,
    rebounds_per_game REAL NOT NULL,
    assists_per_game REAL NOT NULL,
    steals_per_game REAL NOT NULL,
    blocks_per_game REAL NOT NULL,
    turnovers_per_game REAL NOT NULL,
    field_goal_percentage REAL NOT NULL,
    three_point_percentage REAL NOT NULL,
    free_throw_percentage REAL NOT NULL,
    PRIMARY KEY (player_id, season),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS player_shot_profiles (
    player_id INTEGER NOT NULL,
    season TEXT NOT NULL,
    shot_attempts INTEGER NOT NULL,
    shot_makes INTEGER NOT NULL,
    three_point_attempts INTEGER NOT NULL,
    three_point_makes INTEGER NOT NULL,
    rim_frequency REAL NOT NULL,
    paint_frequency REAL NOT NULL,
    midrange_frequency REAL NOT NULL,
    three_point_frequency REAL NOT NULL,
    dunk_frequency REAL NOT NULL,
    layup_frequency REAL NOT NULL,
    floater_frequency REAL NOT NULL,
    hook_frequency REAL NOT NULL,
    pull_up_frequency REAL NOT NULL,
    step_back_frequency REAL NOT NULL,
    average_shot_distance REAL NOT NULL,
    shot_difficulty_index REAL NOT NULL,
    shot_making_above_expected REAL NOT NULL,
    PRIMARY KEY (player_id, season),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS data_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_games_player_date
    ON player_game_stats(player_id, game_date DESC);
"""

SHOT_PROFILE_MIGRATIONS = {
    "shot_makes": "INTEGER NOT NULL DEFAULT 0",
    "three_point_attempts": "INTEGER NOT NULL DEFAULT 0",
    "three_point_makes": "INTEGER NOT NULL DEFAULT 0",
}


def get_connection(database_path: Path | str | None = None) -> sqlite3.Connection:
    """Return a configured SQLite connection."""

    path = Path(database_path or settings.database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def database_session(
    database_path: Path | str | None = None,
) -> Iterator[sqlite3.Connection]:
    """Commit a transaction on success and roll it back on failure."""

    connection = get_connection(database_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(database_path: Path | str | None = None) -> None:
    """Create the application schema if it does not exist."""

    with database_session(database_path) as connection:
        connection.executescript(SCHEMA_SQL)
        existing = {
            row[1]
            for row in connection.execute("PRAGMA table_info(player_shot_profiles)")
        }
        for column, definition in SHOT_PROFILE_MIGRATIONS.items():
            if column not in existing:
                connection.execute(
                    f"ALTER TABLE player_shot_profiles ADD COLUMN {column} {definition}"
                )


def _insert_frame(
    connection: sqlite3.Connection, table: str, frame: pd.DataFrame
) -> None:
    """Insert a frame using the active SQLite transaction."""

    columns = list(frame.columns)
    names = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    connection.executemany(
        f"INSERT INTO {table} ({names}) VALUES ({placeholders})",
        frame.itertuples(index=False, name=None),
    )


def replace_dataset(
    players: pd.DataFrame,
    season_stats: pd.DataFrame,
    game_stats: pd.DataFrame,
    metadata: dict[str, str],
    database_path: Path | str | None = None,
    shot_profiles: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Atomically replace analytics data and its provenance metadata."""

    initialize_database(database_path)
    with database_session(database_path) as connection:
        connection.execute("DELETE FROM player_game_stats")
        connection.execute("DELETE FROM player_shot_profiles")
        connection.execute("DELETE FROM player_season_stats")
        connection.execute("DELETE FROM players")
        connection.execute("DELETE FROM data_metadata")
        _insert_frame(connection, "players", players)
        _insert_frame(connection, "player_season_stats", season_stats)
        _insert_frame(connection, "player_game_stats", game_stats)
        if shot_profiles is not None and not shot_profiles.empty:
            _insert_frame(connection, "player_shot_profiles", shot_profiles)
        connection.executemany(
            "INSERT INTO data_metadata(key, value) VALUES (?, ?)",
            [(str(key), str(value)) for key, value in metadata.items()],
        )
    return {
        "players": len(players),
        "season_rows": len(season_stats),
        "game_rows": len(game_stats),
        "shot_profile_rows": 0 if shot_profiles is None else len(shot_profiles),
    }


def get_data_metadata(database_path: Path | str | None = None) -> dict[str, str]:
    """Return provenance for the currently loaded dataset."""

    initialize_database(database_path)
    frame = _read_sql("SELECT key, value FROM data_metadata", database_path=database_path)
    return dict(zip(frame["key"], frame["value"])) if not frame.empty else {}


def get_data_status_text(database_path: Path | str | None = None) -> str:
    """Create a concise source statement suitable for AI evidence packets."""

    metadata = get_data_metadata(database_path)
    if metadata.get("provider_kind") == "kaggle_cc0_snapshot":
        status = (
            f"CC0-licensed Kaggle snapshot for {metadata.get('season', 'unknown season')}, "
            f"dataset version {metadata.get('dataset_version', 'unknown')}, retrieved "
            f"{metadata.get('retrieved_at', 'at an unknown time')}."
        )
        if metadata.get("shot_data_provider"):
            status += (
                " Shot-style features were engineered from a stored NBA shot-detail "
                f"snapshot retrieved {metadata.get('shot_data_retrieved_at', 'at an unknown time')}."
            )
        return status
    return "Bundled offline demo snapshot; recent game rows are illustrative."


def _read_sql(query: str, params: tuple = (), database_path: Path | str | None = None) -> pd.DataFrame:
    with get_connection(database_path) as connection:
        return pd.read_sql_query(query, connection, params=params)


def get_players(database_path: Path | str | None = None) -> pd.DataFrame:
    """Return searchable player metadata."""

    return _read_sql(
        "SELECT player_id, player_name, team, position FROM players ORDER BY player_name",
        database_path=database_path,
    )


def get_player_profile(
    player_id: int, database_path: Path | str | None = None
) -> dict | None:
    """Return player metadata plus the latest season row."""

    query = """
        SELECT p.*, s.*,
            sp.shot_makes * 1.0 / NULLIF(s.games_played, 0) AS field_goals_made_per_game,
            sp.shot_attempts * 1.0 / NULLIF(s.games_played, 0) AS field_goal_attempts_per_game,
            sp.three_point_makes * 1.0 / NULLIF(s.games_played, 0) AS three_points_made_per_game,
            sp.three_point_attempts * 1.0 / NULLIF(s.games_played, 0) AS three_point_attempts_per_game,
            (sp.shot_makes - sp.three_point_makes) * 1.0 / NULLIF(s.games_played, 0) AS two_points_made_per_game,
            (sp.shot_attempts - sp.three_point_attempts) * 1.0 / NULLIF(s.games_played, 0) AS two_point_attempts_per_game,
            (sp.shot_makes - sp.three_point_makes) * 1.0 /
                NULLIF(sp.shot_attempts - sp.three_point_attempts, 0) AS two_point_percentage,
            (sp.shot_makes + 0.5 * sp.three_point_makes) * 1.0 /
                NULLIF(sp.shot_attempts, 0) AS effective_field_goal_percentage,
            sp.shot_attempts,
            sp.rim_frequency,
            sp.paint_frequency,
            sp.midrange_frequency,
            sp.three_point_frequency,
            sp.dunk_frequency,
            sp.layup_frequency,
            sp.floater_frequency,
            sp.hook_frequency,
            sp.pull_up_frequency,
            sp.step_back_frequency,
            sp.average_shot_distance,
            sp.shot_difficulty_index,
            sp.shot_making_above_expected
        FROM players p
        JOIN player_season_stats s USING (player_id)
        LEFT JOIN player_shot_profiles sp
            ON sp.player_id = s.player_id AND sp.season = s.season
        WHERE p.player_id = ?
        ORDER BY s.season DESC
        LIMIT 1
    """
    frame = _read_sql(query, (int(player_id),), database_path)
    if not frame.empty and float(frame.iloc[0]["three_point_attempts_per_game"] or 0) == 0:
        frame.loc[:, "three_point_percentage"] = float("nan")
    return None if frame.empty else frame.iloc[0].to_dict()


def get_season_stats(database_path: Path | str | None = None) -> pd.DataFrame:
    """Return the latest season profile for every player."""

    query = """
        SELECT
            p.player_id,
            p.player_name,
            p.team,
            p.position,
            s.season,
            s.games_played,
            s.minutes_per_game,
            s.points_per_game,
            s.rebounds_per_game,
            s.assists_per_game,
            s.steals_per_game,
            s.blocks_per_game,
            s.turnovers_per_game,
            s.field_goal_percentage,
            s.three_point_percentage,
            s.free_throw_percentage
            ,sp.shot_makes * 1.0 / NULLIF(s.games_played, 0) AS field_goals_made_per_game
            ,sp.shot_attempts * 1.0 / NULLIF(s.games_played, 0) AS field_goal_attempts_per_game
            ,sp.three_point_makes * 1.0 / NULLIF(s.games_played, 0) AS three_points_made_per_game
            ,sp.three_point_attempts * 1.0 / NULLIF(s.games_played, 0) AS three_point_attempts_per_game
            ,(sp.shot_makes - sp.three_point_makes) * 1.0 / NULLIF(s.games_played, 0) AS two_points_made_per_game
            ,(sp.shot_attempts - sp.three_point_attempts) * 1.0 / NULLIF(s.games_played, 0) AS two_point_attempts_per_game
            ,(sp.shot_makes - sp.three_point_makes) * 1.0 /
                NULLIF(sp.shot_attempts - sp.three_point_attempts, 0) AS two_point_percentage
            ,(sp.shot_makes + 0.5 * sp.three_point_makes) * 1.0 /
                NULLIF(sp.shot_attempts, 0) AS effective_field_goal_percentage
            ,sp.shot_attempts
            ,sp.rim_frequency
            ,sp.paint_frequency
            ,sp.midrange_frequency
            ,sp.three_point_frequency
            ,sp.dunk_frequency
            ,sp.layup_frequency
            ,sp.floater_frequency
            ,sp.hook_frequency
            ,sp.pull_up_frequency
            ,sp.step_back_frequency
            ,sp.average_shot_distance
            ,sp.shot_difficulty_index
            ,sp.shot_making_above_expected
        FROM players p
        JOIN player_season_stats s USING (player_id)
        LEFT JOIN player_shot_profiles sp
            ON sp.player_id = s.player_id AND sp.season = s.season
        WHERE s.season = (SELECT MAX(season) FROM player_season_stats)
        ORDER BY p.player_name
    """
    frame = _read_sql(query, database_path=database_path)
    frame.loc[
        pd.to_numeric(frame["three_point_attempts_per_game"], errors="coerce").fillna(0).eq(0),
        "three_point_percentage",
    ] = float("nan")
    return frame


def get_player_shot_profile(
    player_id: int, database_path: Path | str | None = None
) -> dict | None:
    """Return the latest engineered shot-style profile for one player."""

    frame = _read_sql(
        """
        SELECT * FROM player_shot_profiles
        WHERE player_id = ?
        ORDER BY season DESC
        LIMIT 1
        """,
        (int(player_id),),
        database_path,
    )
    return None if frame.empty else frame.iloc[0].to_dict()


def get_recent_games(
    player_id: int,
    limit: int = 10,
    database_path: Path | str | None = None,
) -> pd.DataFrame:
    """Return a player's newest games in chronological order."""

    query = """
        SELECT * FROM (
            SELECT * FROM player_game_stats
            WHERE player_id = ?
            ORDER BY game_date DESC
            LIMIT ?
        ) ORDER BY game_date ASC
    """
    return _read_sql(query, (int(player_id), int(limit)), database_path)
