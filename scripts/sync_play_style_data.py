"""Download recent matchup, action, and optional possession files."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DATASET = "https://huggingface.co/datasets/cdechoch/nba-data-archive/resolve/main/per_season"


def download(url: str, destination: Path, timeout: int = 120) -> None:
    """Download one file atomically, resuming an interrupted transfer."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        mode = "ab" if downloaded and response.status_code == 206 else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-season", type=int, default=2017)
    parser.add_argument("--last-season", type=int, default=2025)
    parser.add_argument(
        "--include-actions",
        action="store_true",
        help="Also download identified NBA CDN play-by-play files.",
    )
    parser.add_argument(
        "--include-possessions",
        action="store_true",
        help="Also download pbpstats possession files (available through 2024-25).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.first_season > args.last_season:
        parser.error("--first-season must not be later than --last-season")

    kinds = [("matchups", "matchup_events", args.last_season)]
    if args.include_actions:
        kinds.append(("cdnnba", "play_by_play_cdn", args.last_season))
    if args.include_possessions:
        kinds.append(("pbpstats", "possession_events", min(args.last_season, 2024)))
    for source, directory, last_season in kinds:
        for season in range(args.first_season, last_season + 1):
            destination = ROOT / "data" / "historical" / directory / f"{season}.parquet"
            if destination.exists() and not args.force:
                print(f"Already present: {destination.name} ({source})")
                continue
            print(f"Downloading {source} {season}-{str(season + 1)[-2:]}...")
            download(
                f"{DATASET}/{source}/{season}.parquet?download=true",
                destination,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
