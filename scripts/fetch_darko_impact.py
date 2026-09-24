"""Snapshot the public nbarapm mirror of full DARKO DPM history."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "historical" / "darko_dpm.parquet"
SOURCE_URL = "https://nbarapm.com/load/DARKO"
SOURCE_PAGE = "https://nbarapm.com/datasets/MetricHistory"


def _normalize_history(records: list[dict[str, object]]) -> pd.DataFrame:
    """Validate and normalize the mirror payload into one row per player-season."""

    frame = pd.DataFrame(records).rename(columns={"nba_id": "player_id"})
    required = {"player_id", "player_name", "season", "dpm", "o_dpm", "d_dpm"}
    if missing := required - set(frame):
        raise ValueError(f"DARKO history is missing columns: {sorted(missing)}")
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    end_year = pd.to_numeric(frame["season"], errors="coerce")
    frame["season"] = end_year.map(
        lambda year: f"{int(year)-1}-{str(int(year))[-2:]}" if pd.notna(year) else None
    )
    numeric = [
        "dpm", "o_dpm", "d_dpm", "box_odpm", "box_ddpm",
        "on_off_odpm", "on_off_ddpm", "dpm_rank", "o_dpm_rank", "d_dpm_rank",
    ]
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if {"box_odpm", "box_ddpm"} <= set(frame):
        frame["box_dpm"] = frame["box_odpm"] + frame["box_ddpm"]
    if {"on_off_odpm", "on_off_ddpm"} <= set(frame):
        frame["on_off_dpm"] = frame["on_off_odpm"] + frame["on_off_ddpm"]
    columns = [
        "season", "player_id", "player_name", "team_name", "dpm", "o_dpm", "d_dpm",
        "box_dpm", "box_odpm", "box_ddpm", "on_off_dpm", "on_off_odpm", "on_off_ddpm",
        "dpm_rank", "o_dpm_rank", "d_dpm_rank",
    ]
    return (
        frame[[column for column in columns if column in frame]]
        .dropna(subset=["season", "player_id"])
        .assign(player_id=lambda value: value["player_id"].astype(int))
        .drop_duplicates(["season", "player_id"], keep="last")
        .sort_values(["season", "player_id"])
        .reset_index(drop=True)
    )


def fetch_history() -> pd.DataFrame:
    """Download all publicly exposed historical DARKO player-seasons."""

    response = requests.get(
        SOURCE_URL, headers={"User-Agent": "BallDNA-AI/portfolio"}, timeout=60
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("DARKO history endpoint did not return a list.")
    return _normalize_history(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ratings = fetch_history()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(args.output, index=False, compression="snappy")
    metadata = {
        "source": SOURCE_PAGE,
        "download_url": SOURCE_URL,
        "metric": "DARKO Daily Plus-Minus",
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seasons": sorted(ratings["season"].unique()),
        "rows": len(ratings),
        "note": (
            "Public nbarapm mirror snapshot used with DARKO attribution; historical team_name "
            "values may use current franchise names, so BallDNA joins impact by player_id and season."
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(ratings):,} DARKO rows to {args.output}")


if __name__ == "__main__":
    main()
