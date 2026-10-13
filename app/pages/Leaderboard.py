"""Public similarity-leaderboard route."""

from pathlib import Path
from runpy import run_path


run_path(
    Path(__file__).parents[1] / "main.py",
    init_globals={"ROUTE_OVERRIDE": "leaderboard"},
)
