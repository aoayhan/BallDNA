"""Normalize the CC0 NBA box-score dataset published by Eoin A Moore on Kaggle."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ball_ai.data.ingest import BasketballDataset, validate_dataset


DATASET_REF = "eoinamoore/historical-nba-data-and-player-box-scores"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_REF}"

TEAM_ABBREVIATIONS = {
    1610612737: "ATL", 1610612738: "BOS", 1610612751: "BKN", 1610612766: "CHA",
    1610612741: "CHI", 1610612739: "CLE", 1610612742: "DAL", 1610612743: "DEN",
    1610612765: "DET", 1610612744: "GSW", 1610612745: "HOU", 1610612754: "IND",
    1610612746: "LAC", 1610612747: "LAL", 1610612763: "MEM", 1610612748: "MIA",
    1610612749: "MIL", 1610612750: "MIN", 1610612740: "NOP", 1610612752: "NYK",
    1610612760: "OKC", 1610612753: "ORL", 1610612755: "PHI", 1610612756: "PHX",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS", 1610612761: "TOR",
    1610612762: "UTA", 1610612764: "WAS",
}

GAME_USECOLS = [
    "firstName", "lastName", "personId", "gameId", "gameDate", "gameType",
    "numMinutes", "points", "assists", "blocks", "steals", "turnovers",
    "reboundsTotal", "fieldGoalsAttempted", "fieldGoalsMade",
    "threePointersAttempted", "threePointersMade", "freeThrowsAttempted",
    "freeThrowsMade", "playerteamId", "opponentteamId",
]


class KaggleDatasetError(RuntimeError):
    """Raised when the licensed raw dataset cannot be normalized safely."""


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.Series(
        np.divide(
            numerator.to_numpy(dtype=float),
            denominator.to_numpy(dtype=float),
            out=np.zeros(len(numerator), dtype=float),
            where=denominator.to_numpy(dtype=float) != 0,
        ),
        index=numerator.index,
    )


def _position(row: pd.Series) -> str:
    flags = (("guard", "G"), ("forward", "F"), ("center", "C"))
    labels = [label for flag, label in flags if row.get(flag) == 1]
    return "-".join(labels) or "N/A"


class KaggleCC0Provider:
    """Build a compact season snapshot from the public CC0 Kaggle files."""

    def __init__(self, recent_games: int = 10, minimum_games: int = 1) -> None:
        self.recent_games = recent_games
        self.minimum_games = minimum_games

    def read_season_games(
        self,
        statistics_path: Path,
        season: str,
        chunk_size: int = 200_000,
    ) -> pd.DataFrame:
        """Stream the historical CSV and keep one regular season in memory."""

        try:
            start_year = int(season.split("-")[0])
        except (ValueError, IndexError) as exc:
            raise KaggleDatasetError("Season must look like 2025-26.") from exc
        start_date = f"{start_year}-09-01"
        end_date = f"{start_year + 1}-07-01"
        selected = []
        try:
            chunks = pd.read_csv(
                statistics_path,
                usecols=GAME_USECOLS,
                chunksize=chunk_size,
                low_memory=False,
            )
            for chunk in chunks:
                dates = chunk["gameDate"].astype(str).str[:10]
                mask = (
                    chunk["gameType"].eq("Regular Season")
                    & dates.ge(start_date)
                    & dates.lt(end_date)
                )
                if mask.any():
                    selected.append(chunk.loc[mask].copy())
        except (FileNotFoundError, ValueError) as exc:
            raise KaggleDatasetError(f"Could not read {statistics_path}: {exc}") from exc
        if not selected:
            raise KaggleDatasetError(f"No regular-season rows found for {season}.")
        games = pd.concat(selected, ignore_index=True)
        games["game_date"] = pd.to_datetime(games["gameDate"], errors="raise").dt.date.astype(str)
        games["numMinutes"] = pd.to_numeric(games["numMinutes"], errors="coerce")
        return games.loc[games["numMinutes"].fillna(0) > 0].copy()

    def normalize(
        self,
        raw_games: pd.DataFrame,
        raw_players: pd.DataFrame,
        season: str,
        dataset_version: str,
        dataset_updated_at: str,
    ) -> BasketballDataset:
        """Aggregate season profiles and retain the latest game rows per player."""

        required_games = set(GAME_USECOLS) | {"game_date"}
        required_players = {"personId", "guard", "forward", "center"}
        if missing := required_games - set(raw_games.columns):
            raise KaggleDatasetError(f"PlayerStatistics is missing columns: {sorted(missing)}")
        if missing := required_players - set(raw_players.columns):
            raise KaggleDatasetError(f"Players is missing columns: {sorted(missing)}")

        games = raw_games.copy()
        numeric_columns = [
            "personId", "gameId", "numMinutes", "points", "assists", "blocks", "steals",
            "turnovers", "reboundsTotal", "fieldGoalsAttempted", "fieldGoalsMade",
            "threePointersAttempted", "threePointersMade", "freeThrowsAttempted",
            "freeThrowsMade", "playerteamId", "opponentteamId",
        ]
        for column in numeric_columns:
            games[column] = pd.to_numeric(games[column], errors="coerce").fillna(0)
        games["personId"] = games["personId"].astype(int)
        games = games.sort_values(["personId", "game_date", "gameId"])
        games = games.drop_duplicates(["personId", "gameId"], keep="last")

        game_counts = games.groupby("personId")["gameId"].nunique()
        eligible_ids = set(game_counts[game_counts >= self.minimum_games].index.astype(int))
        games = games.loc[games["personId"].isin(eligible_ids)].copy()
        if games.empty:
            raise KaggleDatasetError("No players passed the minimum-games filter.")

        grouped = games.groupby("personId", sort=False)
        season_stats = grouped.agg(
            games_played=("gameId", "nunique"),
            minutes_per_game=("numMinutes", "mean"),
            points_per_game=("points", "mean"),
            rebounds_per_game=("reboundsTotal", "mean"),
            assists_per_game=("assists", "mean"),
            steals_per_game=("steals", "mean"),
            blocks_per_game=("blocks", "mean"),
            turnovers_per_game=("turnovers", "mean"),
            field_goals_made=("fieldGoalsMade", "sum"),
            field_goal_attempts=("fieldGoalsAttempted", "sum"),
            three_points_made=("threePointersMade", "sum"),
            three_point_attempts=("threePointersAttempted", "sum"),
            free_throws_made=("freeThrowsMade", "sum"),
            free_throw_attempts=("freeThrowsAttempted", "sum"),
        ).reset_index(names="player_id")
        season_stats["field_goal_percentage"] = _safe_ratio(
            season_stats.pop("field_goals_made"), season_stats.pop("field_goal_attempts")
        )
        season_stats["three_point_percentage"] = _safe_ratio(
            season_stats.pop("three_points_made"), season_stats.pop("three_point_attempts")
        )
        season_stats["free_throw_percentage"] = _safe_ratio(
            season_stats.pop("free_throws_made"), season_stats.pop("free_throw_attempts")
        )
        season_stats.insert(1, "season", season)
        season_stats["games_played"] = season_stats["games_played"].astype(int)

        latest = games.sort_values("game_date").groupby("personId", as_index=False).tail(1)
        players = latest[["personId", "firstName", "lastName", "playerteamId"]].copy()
        players["player_id"] = players.pop("personId").astype(int)
        players["player_name"] = (
            players.pop("firstName").fillna("").str.strip()
            + " "
            + players.pop("lastName").fillna("").str.strip()
        ).str.strip()
        players["team"] = players.pop("playerteamId").map(TEAM_ABBREVIATIONS).fillna("N/A")
        player_positions = (
            raw_players.drop_duplicates("personId")
            .set_index("personId")
            .apply(_position, axis=1)
        )
        players["position"] = players["player_id"].map(player_positions).fillna("N/A")
        players = players[["player_id", "player_name", "team", "position"]]

        recent = (
            games.sort_values(["personId", "game_date", "gameId"], ascending=[True, False, False])
            .groupby("personId", group_keys=False)
            .head(self.recent_games)
            .copy()
        )
        game_stats = pd.DataFrame(
            {
                "game_id": recent["gameId"].round().astype(int).astype(str).str.zfill(10),
                "player_id": recent["personId"].astype(int),
                "game_date": recent["game_date"],
                "opponent": recent["opponentteamId"].map(TEAM_ABBREVIATIONS).fillna("N/A"),
                "minutes": recent["numMinutes"].round(1),
                "points": recent["points"].round().astype(int),
                "rebounds": recent["reboundsTotal"].round().astype(int),
                "assists": recent["assists"].round().astype(int),
                "steals": recent["steals"].round().astype(int),
                "blocks": recent["blocks"].round().astype(int),
                "turnovers": recent["turnovers"].round().astype(int),
                "field_goal_attempts": recent["fieldGoalsAttempted"].round().astype(int),
                "field_goals_made": recent["fieldGoalsMade"].round().astype(int),
                "three_point_attempts": recent["threePointersAttempted"].round().astype(int),
                "three_points_made": recent["threePointersMade"].round().astype(int),
                "free_throw_attempts": recent["freeThrowsAttempted"].round().astype(int),
                "free_throws_made": recent["freeThrowsMade"].round().astype(int),
            }
        )

        retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        dataset = BasketballDataset(
            players=players.sort_values("player_name").reset_index(drop=True),
            season_stats=season_stats.reset_index(drop=True),
            game_stats=game_stats.reset_index(drop=True),
            metadata={
                "provider": "Eoin A Moore NBA Dataset on Kaggle",
                "provider_kind": "kaggle_cc0_snapshot",
                "license": "CC0-1.0",
                "season": season,
                "retrieved_at": retrieved_at,
                "dataset_version": str(dataset_version),
                "dataset_updated_at": str(dataset_updated_at),
                "source_url": DATASET_URL,
                "recent_game_provenance": "Kaggle PlayerStatistics.csv, latest games selected locally",
                "position_note": "Position derived from the dataset's guard/forward/center flags.",
                "upstream_note": "The dataset publisher attributes the underlying statistics to NBA.com.",
            },
        )
        validate_dataset(dataset)
        return dataset

    def from_files(
        self,
        statistics_path: Path,
        players_path: Path,
        season: str,
        dataset_version: str,
        dataset_updated_at: str,
    ) -> BasketballDataset:
        raw_games = self.read_season_games(statistics_path, season)
        raw_players = pd.read_csv(players_path, low_memory=False)
        return self.normalize(
            raw_games, raw_players, season, dataset_version, dataset_updated_at
        )
