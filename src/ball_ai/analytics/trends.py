"""Recent-vs-prior game trend calculations."""

from __future__ import annotations

import pandas as pd

from ball_ai.analytics.metrics import add_game_metrics


TREND_METRICS = {
    "effective_field_goal_percentage": "eFG%",
    "rebounds": "Rebounds",
    "assists": "Assists",
    "points": "Points",
}


def calculate_trends(games: pd.DataFrame, window: int = 5) -> list[dict]:
    """Compare the latest window with the preceding window.

    At least six rows are required so each side contains three games.
    """

    if len(games) < 6:
        return []
    enriched = add_game_metrics(games).sort_values("game_date")
    effective_window = min(window, len(enriched) // 2)
    prior = enriched.iloc[-2 * effective_window : -effective_window]
    recent = enriched.iloc[-effective_window:]
    output = []
    for column, label in TREND_METRICS.items():
        prior_average = float(prior[column].mean())
        recent_average = float(recent[column].mean())
        change = recent_average - prior_average
        tolerance = 0.01 if "percentage" in column else 0.25
        direction = "up" if change > tolerance else "down" if change < -tolerance else "stable"
        output.append(
            {
                "metric": column,
                "label": label,
                "recent_average": round(recent_average, 3),
                "prior_average": round(prior_average, 3),
                "change": round(change, 3),
                "direction": direction,
                "window": effective_window,
            }
        )
    return output


def trend_sentence(trends: list[dict]) -> str:
    """Turn calculated movements into a cautious one-sentence summary."""

    if not trends:
        return "There are not enough recent games to calculate a reliable split trend."
    meaningful = [item for item in trends if item["direction"] != "stable"]
    if not meaningful:
        return "The measured recent indicators are broadly stable across the two windows."
    fragments = [
        f"{item['label']} is {item['direction']} from {item['prior_average']:.1f} to {item['recent_average']:.1f}"
        if item["metric"] != "effective_field_goal_percentage"
        else f"{item['label']} is {item['direction']} from {item['prior_average']:.1%} to {item['recent_average']:.1%}"
        for item in meaningful
    ]
    return "; ".join(fragments) + "."
