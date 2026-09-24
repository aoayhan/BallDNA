"""Repair missing player-team assignments from the official NBA game-log endpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests


URL = "https://stats.nba.com/stats/leaguegamelog"
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.nba.com/",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
    "Sec-Ch-Ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Fetch-Dest": "empty",
}


def fetch_assignments(season: str, timeout: int = 120) -> pd.DataFrame:
    params = {
        "Counter": 0, "Direction": "ASC", "LeagueID": "00",
        "PlayerOrTeam": "P", "Season": season,
        "SeasonType": "Regular Season", "Sorter": "DATE",
        "DateFrom": "", "DateTo": "",
    }
    for attempt in range(3):
        try:
            response = requests.get(URL, params=params, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    result = response.json()["resultSets"][0]
    frame = pd.DataFrame(result["rowSet"], columns=result["headers"])
    frame["game_id"] = frame["GAME_ID"].astype(str).str.zfill(10)
    frame["player_id"] = frame["PLAYER_ID"].astype(int)
    frame["team_repair"] = frame["TEAM_ABBREVIATION"].astype(str)
    frame["opponent_repair"] = frame["MATCHUP"].str.extract(r"([A-Z]{3})$")
    return frame[["game_id", "player_id", "team_repair", "opponent_repair"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2021-22")
    parser.add_argument("--archive", type=Path, default=Path("data/historical"))
    args = parser.parse_args()

    path = args.archive / "player_game_stats" / f"season={args.season}" / "season_type=regular" / "part-0000.parquet"
    games = pd.read_parquet(path)
    games["game_id"] = games["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    repaired = games.merge(
        fetch_assignments(args.season),
        on=["game_id", "player_id"],
        how="left",
        validate="one_to_one",
    )
    matched = repaired["team_repair"].notna()
    if matched.mean() < 0.95:
        raise RuntimeError(f"Only {matched.mean():.1%} of player-game rows matched NBA Stats.")
    repaired.loc[matched, "team"] = repaired.loc[matched, "team_repair"]
    repaired.loc[matched, "opponent"] = repaired.loc[matched, "opponent_repair"]
    repaired = repaired.drop(columns=["team_repair", "opponent_repair"])
    temporary = path.with_suffix(".tmp.parquet")
    repaired.to_parquet(temporary, index=False, compression="snappy")
    temporary.replace(path)
    metadata_path = args.archive / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.setdefault("sources", {})[
            f"player_team_assignments_{args.season}"
        ] = URL
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Repaired {matched.sum():,}/{len(repaired):,} {args.season} player-game rows.")


if __name__ == "__main__":
    main()
