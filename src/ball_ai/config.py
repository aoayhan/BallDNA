"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def _path_from_env(name: str, default: Path) -> Path:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    path = Path(raw_value).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


@dataclass(frozen=True)
class Settings:
    """Runtime settings with safe local defaults."""

    root_dir: Path = ROOT_DIR
    sample_data_dir: Path = ROOT_DIR / "data" / "sample"
    nba_snapshot_dir: Path = ROOT_DIR / "data" / "nba_snapshot"
    historical_data_dir: Path = _path_from_env(
        "HOOPLENS_HISTORICAL_DATA_DIR", ROOT_DIR / "data" / "historical"
    )
    database_path: Path = _path_from_env(
        "HOOPLENS_DB_PATH", ROOT_DIR / "data" / "hooplens.db"
    )
    nba_season: str = os.getenv("HOOPLENS_NBA_SEASON", "2025-26")
    minimum_games: int = int(os.getenv("HOOPLENS_MIN_GAMES", "1"))
    recent_games: int = int(os.getenv("HOOPLENS_RECENT_GAMES", "10"))


settings = Settings()
