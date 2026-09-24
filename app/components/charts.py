"""Plotly charts used by multiple app pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


ORANGE = "#FF6B35"
BLUE = "#47A7FF"
SIMILARITY_BUCKETS = (
    (0.00, 0.60, "Unlike", "gray"),
    (0.60, 0.70, "Some shared tendencies", "gray"),
    (0.70, 0.80, "Partial style match", "blue"),
    (0.80, 0.85, "Some overlap in style", "blue"),
    (0.85, 0.90, "Close style match", "orange"),
    (0.90, 0.94, "Style twin", "green"),
    (0.94, 1.01, "Basketball doppelgänger", "violet"),
)


def similarity_bucket(score: float) -> str:
    return next(
        label for lower, upper, label, _ in SIMILARITY_BUCKETS
        if lower <= score < upper
    )


def recent_box_score_chart(games: pd.DataFrame) -> go.Figure:
    long = games.melt(
        id_vars=["game_date", "opponent"],
        value_vars=["rebounds", "assists", "points"],
        var_name="Metric",
        value_name="Value",
    )
    figure = px.line(
        long,
        x="game_date",
        y="Value",
        color="Metric",
        markers=True,
        color_discrete_map={"points": ORANGE, "rebounds": BLUE, "assists": "#64D6A4"},
    )
    figure.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        legend_title_text="",
        xaxis_title="Game date",
        hovermode="x unified",
    )
    return figure


def shooting_trend_chart(games: pd.DataFrame) -> go.Figure:
    frame = games.copy()
    frame["FG%"] = frame["field_goals_made"].div(frame["field_goal_attempts"]).fillna(0)
    frame["3P%"] = frame["three_points_made"].div(frame["three_point_attempts"]).fillna(0)
    long = frame.melt(
        id_vars="game_date", value_vars=["FG%", "3P%"], var_name="Metric", value_name="Rate"
    )
    figure = px.line(long, x="game_date", y="Rate", color="Metric", markers=True)
    figure.update_yaxes(tickformat=".0%", range=[0, max(.8, float(long["Rate"].max()) + .05)])
    figure.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
    return figure


def comparison_bar_chart(first: dict, second: dict) -> go.Figure:
    metrics = {
        "MP": "minutes_per_game",
        "FG": "field_goals_made_per_game",
        "FGA": "field_goal_attempts_per_game",
        "3P": "three_points_made_per_game",
        "3PA": "three_point_attempts_per_game",
        "2P": "two_points_made_per_game",
        "2PA": "two_point_attempts_per_game",
        "TRB": "rebounds_per_game",
        "AST": "assists_per_game",
        "STL": "steals_per_game",
        "BLK": "blocks_per_game",
        "TOV": "turnovers_per_game",
        "PTS": "points_per_game",
        "DPM": "dpm",
        "O-DPM": "o_dpm",
    }
    figure = go.Figure()
    for profile, color in ((first, ORANGE), (second, BLUE)):
        figure.add_bar(
            name=profile["player_name"],
            x=list(metrics),
            y=[profile[key] for key in metrics.values()],
            marker_color=color,
        )
    figure.update_layout(
        barmode="group",
        dragmode="pan",
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="Per game / DARKO impact",
    )
    figure.update_xaxes(range=[-.5, 7.5], fixedrange=False)
    figure.update_yaxes(zeroline=True)
    return figure


def similarity_chart(frame: pd.DataFrame, impact_label: str | None = None) -> go.Figure:
    ordered = frame.sort_values("similarity_score", ascending=False).copy()
    ordered["similarity_bucket"] = ordered["similarity_score"].map(similarity_bucket)
    ordered["chart_label"] = ordered["player_name"]
    if "season" in ordered:
        ordered["chart_label"] = (
            ordered["player_name"] + " · " + ordered["season"].astype(str)
        )
    figure = px.bar(
        ordered,
        x="similarity_score",
        y="chart_label",
        orientation="h",
        color="similarity_score",
        color_continuous_scale=[[0, "#27344a"], [1, ORANGE]],
        range_x=[0, 1],
        category_orders={"chart_label": ordered["chart_label"].tolist()},
    )
    hover_template = "Similarity: %{x:.2%}<br>Band: %{customdata[0]}"
    if impact_label and "impact_difference" in ordered:
        ordered["impact_direction"] = ordered["impact_difference"].map(
            lambda value: "higher" if value > 0 else "lower" if value < 0 else "equal"
        )
        ordered["impact_text"] = ordered["impact_difference"].map(
            lambda value: f"{value:+.2f} {impact_label}"
        )
        figure.update_traces(
            customdata=ordered[
                ["similarity_bucket", "impact_difference", "impact_direction"]
            ].to_numpy(),
            text=ordered["impact_text"],
            textposition="inside",
            insidetextanchor="end",
            hovertemplate=(
                hover_template
                + f"<br>{impact_label} vs reference: "
                + "%{customdata[1]:+.2f} (%{customdata[2]})<extra></extra>"
            ),
        )
    else:
        figure.update_traces(
            customdata=ordered[["similarity_bucket"]].to_numpy(),
            hovertemplate=hover_template + "<extra></extra>",
        )
    figure.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis_showscale=False,
        xaxis_title=(
            "Similarity index · Higher = more similar basketball tendencies, "
            "not equal ability, quality, or probability"
        ),
        yaxis_title="",
    )
    figure.update_xaxes(
        tickformat=".0%",
        dtick=.05,
        showgrid=True,
        gridcolor="rgba(100,116,139,.20)",
        zeroline=False,
    )
    return figure


def similarity_radar_chart(
    contribution_frame: pd.DataFrame,
    reference_name: str,
    match_name: str,
) -> go.Figure:
    """Compare two player embeddings on league-percentile axes."""

    frame = contribution_frame.iloc[::-1]
    categories = frame["Feature"].tolist()
    reference = (frame["Reference percentile"] * 100).tolist()
    match = (frame["Match percentile"] * 100).tolist()
    if categories:
        categories = [*categories, categories[0]]
        reference = [*reference, reference[0]]
        match = [*match, match[0]]
    figure = go.Figure()
    figure.add_trace(
        go.Scatterpolar(
            r=reference,
            theta=categories,
            fill="toself",
            name=reference_name,
            line_color=ORANGE,
            opacity=0.72,
        )
    )
    figure.add_trace(
        go.Scatterpolar(
            r=match,
            theta=categories,
            fill="toself",
            name=match_name,
            line_color=BLUE,
            opacity=0.64,
        )
    )
    figure.update_layout(
        height=470,
        margin=dict(l=35, r=35, t=40, b=35),
        polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="%", showline=False)),
        legend=dict(orientation="h", y=-0.08),
    )
    return figure
