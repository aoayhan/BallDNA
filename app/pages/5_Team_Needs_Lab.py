"""Explainable team-gap analysis and player-fit retrieval page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from _bootstrap import initialize_app  # noqa: E402
from components.styles import apply_styles  # noqa: E402
from components.ui import (  # noqa: E402
    render_build_trace,
    render_copyable_table,
)
from ball_ai.ai.grounding import build_team_needs_evidence  # noqa: E402
from ball_ai.ai.report_generator import generate_team_needs_report  # noqa: E402
from ball_ai.analytics.team_needs import (  # noqa: E402
    PERCENT_FEATURES,
    analyze_team_needs,
    load_player_candidate_pool,
    load_team_model_evaluation,
    recommend_consensus_players,
)
from ball_ai.config import settings  # noqa: E402


st.set_page_config(page_title="Team Needs Lab · Basketball DNA", page_icon="🧪", layout="wide")
apply_styles()
initialize_app()
@st.cache_data(show_spinner=False)
def load_team_features() -> pd.DataFrame:
    path = settings.historical_data_dir / "team_season_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_candidates(season: str) -> pd.DataFrame:
    return load_player_candidate_pool(season)


def display_metric(feature: str, value: float) -> str:
    if pd.isna(value):
        return "—"
    if feature in PERCENT_FEATURES or feature == "assist_ratio":
        return f"{float(value):.1%}"
    return f"{float(value):.1f}"


def candidate_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Player": frame["player_name"],
            "Team": frame["team"],
            "Pos": frame["position"],
            "G": frame["games_played"].astype(int),
            "MP": frame["minutes_per_game"].map(lambda value: f"{value:.1f}"),
            "eFG%": frame["effective_field_goal_percentage"].map(
                lambda value: "—" if pd.isna(value) else f"{value:.1%}"
            ),
            "3P%": frame["three_point_percentage"].map(
                lambda value: "—" if pd.isna(value) else f"{value:.1%}"
            ),
            "TRB": frame["rebounds_per_game"].map(lambda value: f"{value:.1f}"),
            "AST": frame["assists_per_game"].map(lambda value: f"{value:.1f}"),
            "PTS": frame["points_per_game"].map(lambda value: f"{value:.1f}"),
            "Fit": frame["fit_score"].map(lambda value: f"{value:.1%}"),
            "Robustness": frame.get(
                "selection_rate", pd.Series(1.0, index=frame.index)
            ).map(lambda value: f"{value:.0%}"),
            "Why": frame["fit_reasons"],
        }
    )


st.title("Team Needs Lab")
st.warning("🚧 Under construction — treat this experimental workflow and its recommendations as provisional.")
st.caption(
    "An AI-first roster-gap system: historical team fingerprints identify measurable needs, "
    "and vector-style trait retrieval finds player examples with auditable scoring."
)

features = load_team_features()
if features.empty:
    st.error(
        "Team-needs assets are missing. Run `python scripts/build_team_needs_assets.py` first."
    )
    st.stop()

available_seasons = sorted(features.loc[features["games"].ge(60), "season"].unique(), reverse=True)
controls = st.columns([1, 1.25, 1, 1])
with controls[0]:
    season = st.selectbox("Season", available_seasons)
season_teams = features.loc[features["season"].eq(season)].sort_values("league_rank")
team_options = {
    f"#{int(row.league_rank)} · {row.team} · {int(row.wins)}-{int(row.games-row.wins)}": int(row.team_id)
    for row in season_teams.itertuples()
}
default_index = min(9, len(team_options) - 1)
with controls[1]:
    team_label = st.selectbox("Team", list(team_options), index=default_index)
with controls[2]:
    top_n = st.slider("Elite cohort", 5, 10, 8, help="Top N teams by league-wide win percentage.")
with controls[3]:
    lookback = st.slider("Benchmark seasons", 3, 8, 5)

analysis = analyze_team_needs(
    features,
    season=season,
    team_id=team_options[team_label],
    top_n=top_n,
    lookback_seasons=lookback,
)
team = analysis["team"]
players = load_candidates(season)
best_candidates = recommend_consensus_players(
    players,
    features,
    season=season,
    team_id=int(team["team_id"]),
    top_k=8,
)
rotation_candidates = recommend_consensus_players(
    players,
    features,
    season=season,
    team_id=int(team["team_id"]),
    rotation_only=True,
    top_k=8,
)
evaluation = load_team_model_evaluation()
packet = build_team_needs_evidence(
    analysis, best_candidates, rotation_candidates, evaluation
)

st.subheader(f"{team['team']} · {season}")
metric_columns = st.columns(5)
metric_columns[0].metric("League rank", f"#{int(team['league_rank'])}")
metric_columns[1].metric("Record", f"{int(team['wins'])}-{int(team['games']-team['wins'])}")
metric_columns[2].metric("Offensive rating", f"{team['offensive_rating']:.1f}")
metric_columns[3].metric("Defensive rating", f"{team['defensive_rating']:.1f}")
metric_columns[4].metric("Net rating", f"{team['net_rating']:+.1f}")
st.caption(
    "Ratings are estimated per 100 possessions from the stored team box scores. Lower Defensive "
    "Rating is better. PER is an individual-player statistic, so the lab does not label a team metric ‘defensive PER’."
)

overview_tab, archetype_tab, candidates_tab, ai_tab, method_tab = st.tabs(
    ["Team gaps", "Needed archetype", "Player examples", "Rule-based summary", "Model evidence"]
)

with overview_tab:
    left, right = st.columns([1.15, 1])
    top_gaps = analysis["gaps"].head(8).copy().sort_values("priority")
    with left:
        st.subheader("Model-weighted distance from elite profile")
        gap_chart = px.bar(
            top_gaps,
            x="priority",
            y="label",
            orientation="h",
            color="gap_z",
            color_continuous_scale="OrRd",
            labels={"priority": "Gap priority", "label": "", "gap_z": "Standardized gap"},
        )
        gap_chart.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=15, b=10))
        st.plotly_chart(gap_chart, width="stretch")
    with right:
        st.subheader(f"What top-{top_n} teams repeatedly share")
        common = analysis["commonalities"].head(7)
        render_copyable_table(
            pd.DataFrame(
                {
                    "Feature": common["label"],
                    "Elite direction": common["desired_direction"],
                    "Historical consistency": common["elite_stability"].map(
                        lambda value: f"{value:.0%}"
                    ),
                    "Model weight": common["importance"].map(lambda value: f"{value:.1%}"),
                }
            ),
        )

    gap_display = analysis["gaps"].head(8)
    render_copyable_table(
        pd.DataFrame(
            {
                "Need": gap_display["label"],
                team["team_abbreviation"]: [
                    display_metric(feature, value)
                    for feature, value in zip(gap_display["feature"], gap_display["team_value"])
                ],
                f"Top-{top_n} median": [
                    display_metric(feature, value)
                    for feature, value in zip(gap_display["feature"], gap_display["elite_median"])
                ],
                "Preferred": gap_display["desired_direction"],
                "Player trait": gap_display["player_trait_label"],
            }
        ),
    )

with archetype_tab:
    traits = analysis["archetype_traits"].head(5).copy()
    if traits.empty:
        st.info("No positive gaps were detected against this benchmark.")
    else:
        total = traits["priority"].sum()
        traits["share"] = traits["priority"] / total
        headline = " + ".join(traits.head(3)["player_trait_label"].str.title())
        st.subheader(headline)
        st.write(
            "The archetype is assembled from the team features with the largest model-weighted "
            "gaps. It is a multi-trait target—not a position label or a subjective guess."
        )
        render_copyable_table(
            pd.DataFrame(
                {
                    "Needed trait": traits["player_trait_label"],
                    "Share of top needs": traits["share"].map(lambda value: f"{value:.1%}"),
                }
            ),
        )

with candidates_tab:
    st.warning(
        "These are statistical fit examples, not claims that a player is available, affordable, "
        "healthy, or a realistic trade target. Fit does not include contracts or chemistry."
    )
    st.caption(
        "Robustness is the share of nine top-team/lookback definitions in which the player remains "
        "a leading fit. Results are capped at three examples per broad position so one center archetype "
        "cannot consume the entire list."
    )
    best_tab, rotation_tab = st.tabs(["Best statistical fit", "Rotation-scale examples"])
    with best_tab:
        render_copyable_table(candidate_table(best_candidates))
    with rotation_tab:
        render_copyable_table(candidate_table(rotation_candidates))

with ai_tab:
    st.write(
        "A deterministic template summarizes the fixed commonalities, gap ranking, archetype mapping, "
        "candidate ranking, validation scores, and limitations. It cannot add or reorder players."
    )
    state_key = f"team_needs_report_{season}_{team['team_id']}_{top_n}_{lookback}"
    if st.button("Generate team-needs report", type="primary"):
        with st.spinner("Explaining the fixed model results…"):
            result = generate_team_needs_report(packet)
            st.session_state[state_key] = result
    if state_key in st.session_state:
        result = st.session_state[state_key]
        st.markdown(result.text)
        st.download_button(
            "Download report as Markdown",
            result.text,
            file_name=f"{team['team_abbreviation'].lower()}_{season}_team_needs.md",
            mime="text/markdown",
        )

with method_tab:
    st.subheader("Model selection")
    model_rows = pd.DataFrame(evaluation.get("models", []))
    if not model_rows.empty:
        render_copyable_table(
            pd.DataFrame(
                {
                    "Model": model_rows["model"],
                    "ROC-AUC": model_rows["roc_auc"].map(lambda value: f"{value:.3f}"),
                    "Average precision": model_rows["average_precision"].map(
                        lambda value: f"{value:.3f}"
                    ),
                    "Top-8 precision": model_rows["top_n_precision"].map(
                        lambda value: f"{value:.1%}"
                    ),
                }
            ),
        )
        st.success(
            f"Selected: {evaluation.get('selected_model')} using rolling validation on "
            f"{', '.join(evaluation.get('validation_seasons', []))}."
        )
    st.subheader("Evidence packet used by the summary")
    st.json(packet, expanded=False)

render_build_trace(
    [
        ("Team ETL", "Recover regular-season rows, pair every team game with its opponent, and aggregate possession-adjusted season fingerprints."),
        ("Model selection", "Compare logistic regression, random forest, and gradient boosting on future-season validation; keep the best top-cohort retrieval model."),
        ("Gap analysis", "Compare the selected team with top-N medians and weight standardized gaps by explainable logistic coefficients."),
        ("Player retrieval", "Map priority gaps to transparent player traits and rank candidates by percentile alignment and sample reliability."),
        ("Rule-based summary", "Translate fixed evidence IDs, rankings, validation results, and limitations into a cited template."),
    ]
)
