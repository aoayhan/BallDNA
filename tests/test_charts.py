import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from components.charts import comparison_bar_chart, similarity_bucket, similarity_chart


def test_similarity_bucket_boundaries() -> None:
    assert similarity_bucket(.49) == "Unlike"
    assert similarity_bucket(.59) == "Unlike"
    assert similarity_bucket(.60) == "Some shared tendencies"
    assert similarity_bucket(.70) == "Partial style match"
    assert similarity_bucket(.80) == "Some overlap in style"
    assert similarity_bucket(.85) == "Close style match"
    assert similarity_bucket(.90) == "Style twin"
    assert similarity_bucket(.94) == "Basketball doppelgänger"


def test_comparison_chart_exposes_full_draggable_stat_window() -> None:
    player = {
        "player_name": "Player",
        "minutes_per_game": 30,
        "field_goals_made_per_game": 8,
        "field_goal_attempts_per_game": 16,
        "three_points_made_per_game": 2,
        "three_point_attempts_per_game": 6,
        "two_points_made_per_game": 6,
        "two_point_attempts_per_game": 10,
        "rebounds_per_game": 7,
        "assists_per_game": 5,
        "steals_per_game": 1,
        "blocks_per_game": 1,
        "turnovers_per_game": 2,
        "points_per_game": 20,
        "dpm": 3,
        "o_dpm": 2,
    }

    figure = comparison_bar_chart(player, {**player, "player_name": "Other"})

    assert list(figure.data[0].x) == [
        "MP", "FG", "FGA", "3P", "3PA", "2P", "2PA", "TRB",
        "AST", "STL", "BLK", "TOV", "PTS", "DPM", "O-DPM",
    ]
    assert figure.layout.dragmode == "pan"
    assert tuple(figure.layout.xaxis.range) == (-.5, 7.5)


def test_similarity_chart_keeps_repeated_player_seasons_separate() -> None:
    figure = similarity_chart(pd.DataFrame({
        "player_name": ["Luka Doncic", "Luka Doncic", "Jayson Tatum"],
        "season": ["2024-25", "2023-24", "2024-25"],
        "similarity_score": [.91, .86, .88],
    }))

    assert list(figure.data[0].y) == [
        "Luka Doncic · 2024-25",
        "Jayson Tatum · 2024-25",
        "Luka Doncic · 2023-24",
    ]
    assert figure.layout.yaxis.autorange is None
    assert figure.layout.xaxis.dtick == .05
    assert figure.layout.xaxis.tickformat == ".0%"
    assert figure.data[0].hovertemplate == (
        "Similarity: %{x:.2%}<br>Band: %{customdata[0]}<extra></extra>"
    )


def test_similarity_chart_explains_comparable_player_impact() -> None:
    figure = similarity_chart(pd.DataFrame({
        "player_name": ["Higher", "Lower"],
        "season": ["2024-25", "2024-25"],
        "similarity_score": [.9012, .85],
        "impact_difference": [.75, -.4],
    }), impact_label="O-DPM")

    assert list(figure.data[0].text) == ["+0.75 O-DPM", "-0.40 O-DPM"]
    assert "O-DPM vs reference" in figure.data[0].hovertemplate
    assert list(figure.data[0].customdata[:, 2]) == ["higher", "lower"]
