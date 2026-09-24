# Dataset provenance and license notice

This directory is a small normalized derivative of **NBA Dataset: Box Scores
and Stats (1947 - Today)**, published on Kaggle by Eoin A Moore.

- Dataset page: <https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores>
- License declared by publisher: **CC0-1.0 / Public Domain**
- Snapshot season: `2025-26` regular season
- Eligibility: at least 10 games played
- Recent window: latest 10 games per eligible player
- Raw files used: `PlayerStatistics.csv` and `Players.csv`
- Dataset version and retrieval timestamps: `metadata.json`

The dataset publisher attributes the underlying statistics to NBA.com. HoopLens
records that statement for provenance and does not claim NBA endorsement or
independently warrant the publisher's rights in upstream material.

Refresh with:

```bash
python scripts/sync_nba_data.py --season 2025-26
```

The refresh checks Kaggle's current license metadata before processing. The
large historical source is held in a temporary directory and is not committed.

## Shot-detail supplement

`player_shot_profiles.csv` is a compact locally engineered derivative of the
2025-26 regular-season `shotdetail` Parquet file in the NBA Data Archive mirror.

- Archive: <https://huggingface.co/datasets/cdechoch/nba-data-archive>
- License declared by archive: **Apache-2.0**
- Upstream named by archive: `stats.nba.com`
- Raw event fields used: player ID, make/miss, shot distance, action type, and shot zone
- Raw season rows: 219,160; rows for the 506 application players: 217,892
- Stored output: 506 aggregate player profiles; raw events are not committed
- Download URL, retrieval timestamp, and SHA-256: `metadata.json`

The derived features include shot-zone and action-type frequencies, average
distance, a smoothed expected-FG difficulty proxy, and shot making above that
expectation. The difficulty proxy does not directly measure defender pressure.

Refresh with:

```bash
python scripts/sync_shot_data.py --season 2025-26
```

This project is a local, non-commercial portfolio demonstration and includes
NBA.com attribution. It does not claim NBA endorsement.

## Player-impact supplement

The Roster Construction Lab uses a compact snapshot of public **DARKO Daily
Plus-Minus** leaderboard exports for 2021-22 through 2025-26.

- Source and methodology: <https://www.darko.app/about>
- Fields retained: player ID, player/team labels, DPM, O-DPM, D-DPM, Box DPM,
  and On/Off DPM
- Refresh: `python scripts/fetch_darko_impact.py`

DARKO is credited as the impact-metric author. HoopLens uses the values for a
local, non-commercial portfolio demo and does not claim authorship of DPM.
