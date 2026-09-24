"""Generate deterministic offline demo data for the checked-in sample CSVs.

The season snapshot uses recognizable 2023-24-style player profiles. Recent game
rows are illustrative, generated examples and are not historical game logs.
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "sample"
SEASON = "2023-24"

PROFILES = [
    (1, "Nikola Jokic", "DEN", "C", 79, 34.6, 26.4, 12.4, 9.0, 1.4, 0.9, 3.0, .583, .359, .817),
    (2, "Luka Doncic", "DAL", "G", 70, 37.5, 33.9, 9.2, 9.8, 1.4, 0.5, 4.0, .487, .382, .786),
    (3, "Shai Gilgeous-Alexander", "OKC", "G", 75, 34.0, 30.1, 5.5, 6.2, 2.0, 0.9, 2.2, .535, .353, .874),
    (4, "Jayson Tatum", "BOS", "F", 74, 35.7, 26.9, 8.1, 4.9, 1.0, 0.6, 2.5, .471, .376, .833),
    (5, "Stephen Curry", "GSW", "G", 74, 32.7, 26.4, 4.5, 5.1, 0.7, 0.4, 2.8, .450, .408, .923),
    (6, "LeBron James", "LAL", "F", 71, 35.3, 25.7, 7.3, 8.3, 1.3, 0.5, 3.5, .540, .410, .750),
    (7, "Bam Adebayo", "MIA", "C-F", 71, 34.0, 19.3, 10.4, 3.9, 1.1, 0.9, 2.3, .521, .357, .755),
    (8, "Jalen Brunson", "NYK", "G", 77, 35.4, 28.7, 3.6, 6.7, 0.9, 0.2, 2.4, .479, .401, .847),
    (9, "Tyrese Haliburton", "IND", "G", 69, 32.2, 20.1, 3.9, 10.9, 1.2, 0.7, 2.3, .477, .364, .855),
    (10, "Anthony Edwards", "MIN", "G-F", 79, 35.1, 25.9, 5.4, 5.1, 1.3, 0.5, 3.1, .461, .357, .836),
    (11, "Kevin Durant", "PHX", "F", 75, 37.2, 27.1, 6.6, 5.0, 0.9, 1.2, 3.3, .523, .413, .856),
    (12, "Victor Wembanyama", "SAS", "C-F", 71, 29.7, 21.4, 10.6, 3.9, 1.2, 3.6, 3.7, .465, .325, .796),
]

OPPONENTS = ["BOS", "MIA", "MIL", "PHX", "DEN", "LAL", "NYK", "MIN", "OKC", "DAL"]


def bounded_int(rng: random.Random, mean: float, spread: float, minimum: int = 0) -> int:
    return max(minimum, int(round(rng.gauss(mean, spread))))


def generate_games(profile: tuple, rng: random.Random) -> list[dict]:
    (
        player_id, _, team, _, _, mpg, ppg, rpg, apg, spg, bpg, topg,
        fg_pct, three_pct, ft_pct,
    ) = profile
    rows = []
    trend = rng.choice([-0.08, -0.04, 0.0, 0.05, 0.09])
    for index in range(10):
        trend_factor = 1 + trend * ((index - 4.5) / 4.5)
        minutes = round(max(18.0, rng.gauss(mpg, 2.7)), 1)
        points = bounded_int(rng, ppg * trend_factor, 4.5)
        rebounds = bounded_int(rng, rpg, 2.2)
        assists = bounded_int(rng, apg * (1 + trend / 2), 2.0)
        steals = bounded_int(rng, spg, 0.8)
        blocks = bounded_int(rng, bpg, 0.8)
        turnovers = bounded_int(rng, topg, 1.1)

        three_rate = 0.45 if three_pct >= .39 else 0.33
        fga = max(5, bounded_int(rng, ppg / max(0.7, 2 * fg_pct), 2.5))
        three_attempts = min(fga, bounded_int(rng, fga * three_rate, 1.6))
        three_made = min(three_attempts, bounded_int(rng, three_attempts * three_pct, 1.0))
        two_attempts = fga - three_attempts
        two_pct = min(.70, fg_pct + .08)
        two_made = min(two_attempts, bounded_int(rng, two_attempts * two_pct, 1.4))
        fgm = two_made + three_made
        fta = bounded_int(rng, max(2.0, ppg * .22), 1.8)
        ftm = min(fta, bounded_int(rng, fta * ft_pct, .8))

        # Use scoring implied by makes so every generated row is internally consistent.
        points = 2 * two_made + 3 * three_made + ftm
        game_day = date(2024, 3, 16) + timedelta(days=index * 3)
        opponent = OPPONENTS[(index + player_id) % len(OPPONENTS)]
        if opponent == team:
            opponent = OPPONENTS[(index + player_id + 1) % len(OPPONENTS)]
        rows.append(
            {
                "game_id": f"DEMO-{player_id:02d}-{index + 1:02d}",
                "player_id": player_id,
                "game_date": game_day.isoformat(),
                "opponent": opponent,
                "minutes": minutes,
                "points": points,
                "rebounds": rebounds,
                "assists": assists,
                "steals": steals,
                "blocks": blocks,
                "turnovers": turnovers,
                "field_goal_attempts": fga,
                "field_goals_made": fgm,
                "three_point_attempts": three_attempts,
                "three_points_made": three_made,
                "free_throw_attempts": fta,
                "free_throws_made": ftm,
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = random.Random(24)
    player_rows = [
        dict(zip(["player_id", "player_name", "team", "position"], p[:4]))
        for p in PROFILES
    ]
    season_fields = [
        "player_id", "season", "games_played", "minutes_per_game",
        "points_per_game", "rebounds_per_game", "assists_per_game",
        "steals_per_game", "blocks_per_game", "turnovers_per_game",
        "field_goal_percentage", "three_point_percentage", "free_throw_percentage",
    ]
    season_rows = []
    for p in PROFILES:
        values = [p[0], SEASON, *p[4:]]
        season_rows.append(dict(zip(season_fields, values)))

    game_rows = []
    for profile in PROFILES:
        game_rows.extend(generate_games(profile, rng))

    write_csv(OUTPUT / "players.csv", list(player_rows[0]), player_rows)
    write_csv(OUTPUT / "player_season_stats.csv", season_fields, season_rows)
    write_csv(OUTPUT / "player_game_stats.csv", list(game_rows[0]), game_rows)
    print(f"Generated {len(player_rows)} players and {len(game_rows)} demo games in {OUTPUT}")


if __name__ == "__main__":
    main()

