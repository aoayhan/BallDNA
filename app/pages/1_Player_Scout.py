"""Single-player scouting report page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from _bootstrap import initialize_app  # noqa: E402
from components.charts import recent_box_score_chart, shooting_trend_chart  # noqa: E402
from components.styles import apply_styles  # noqa: E402
from components.ui import (  # noqa: E402
    player_selector,
    render_build_trace,
    render_copyable_table,
    render_evidence,
    render_profile_metrics,
)
from ball_ai.analytics.tables import (  # noqa: E402
    basketball_reference_game_log,
    historical_season_table,
    shot_profile_table,
)
from ball_ai.ai.grounding import build_player_evidence  # noqa: E402
from ball_ai.ai.quality import evaluate_report  # noqa: E402
from ball_ai.ai.report_generator import generate_scouting_report  # noqa: E402
from ball_ai.data.database import (  # noqa: E402
    get_data_status_text,
    get_player_profile,
    get_player_shot_profile,
    get_players,
    get_recent_games,
)
from ball_ai.data.historical_store import get_player_season_history  # noqa: E402


st.set_page_config(page_title="Player Scout · BallDNA", page_icon="🔎", layout="wide")
apply_styles()
initialize_app()

st.title("Player Scout")
st.caption(
    "Create a rule-based scouting summary from a compact, inspectable evidence packet."
)

players = get_players()
player_id = player_selector(
    players,
    "Player",
    "scout_player",
    default_player_name="Shai Gilgeous-Alexander",
)
if player_id is None:
    st.stop()
profile = get_player_profile(player_id)
games = get_recent_games(player_id, limit=10)
shot_profile = get_player_shot_profile(player_id)
season_history = get_player_season_history(player_id)
if profile is None:
    st.error("No season profile is available for this player.")
    st.stop()

st.subheader(f"{profile['player_name']}  ·  {profile['team']}  ·  {profile['position']}")
st.caption(
    "Season statistics run left-to-right in Basketball Reference order; unavailable source fields are omitted."
)
render_profile_metrics(profile)

history_tab, chart_tab, shooting_tab, rows_tab, shot_tab = st.tabs(
    ["Career seasons", "Production trend", "Shooting trend", "Game rows", "Shot fingerprint"]
)
with history_tab:
    if season_history.empty:
        st.info("Historical Parquet data is not installed for this player.")
    else:
        render_copyable_table(historical_season_table(season_history))
        st.caption(
            "Regular-season career rows from the local Parquet archive. Missing early-era "
            "statistics display as —; they are never interpreted as zero."
        )
with chart_tab:
    st.plotly_chart(recent_box_score_chart(games), width="stretch")
with shooting_tab:
    st.plotly_chart(shooting_trend_chart(games), width="stretch")
with rows_tab:
    render_copyable_table(basketball_reference_game_log(games))
with shot_tab:
    if shot_profile:
        render_copyable_table(shot_profile_table(shot_profile))
        st.caption(
            "Supplementary NBA shot-detail features stay separate from the familiar box-score row. "
            "Difficulty is one minus expected FG% for the player's zone/action mix; Shot Making +/- "
            "is actual FG% minus that expectation."
        )
    else:
        st.info("No shot-detail profile is available for this player.")

packet = build_player_evidence(
    profile, games, get_data_status_text(), season_history=season_history
)
st.divider()
controls, note = st.columns([1, 2])
with controls:
    generate = st.button("Create scouting summary", type="primary", width="stretch")
with note:
    st.markdown(
        "<div class='callout'><strong>Evidence contract</strong><br/>The deterministic template uses only "
        "the evidence displayed below—not browsing results or hidden player context.</div>",
        unsafe_allow_html=True,
    )

state_key = f"scout_report_{player_id}"
if generate:
    with st.spinner("Preparing evidence-linked summary…"):
        result = generate_scouting_report(packet)
        st.session_state[state_key] = result

if state_key in st.session_state:
    result = st.session_state[state_key]
    st.markdown(result.text)
    st.download_button(
        "Download report as Markdown",
        data=result.text,
        file_name=f"{profile['player_name'].lower().replace(' ', '_')}_scouting_report.md",
        mime="text/markdown",
    )
    with st.expander("Product quality checks", expanded=False):
        checks = evaluate_report(result.text, packet)
        for name, check in checks.items():
            icon = "✅" if check["passed"] else "⚠️"
            detail = f" · {check['details']}" if check["details"] else ""
            st.write(f"{icon} {name.replace('_', ' ').title()}{detail}")

st.subheader("Evidence used by the summary")
render_evidence(packet)
render_build_trace(
    [
        ("SQL retrieval", "Load one season profile and the player's latest stored game window."),
        ("Feature engineering", "Calculate shooting efficiency, recent aggregates, and recent-versus-prior trend deltas."),
        ("Evidence selection", "Serialize only approved current, recent, and prior-season statistics, provenance, evidence IDs, and explicit limitations."),
        ("Rule-based summary", "Fill a fixed cited template without outside knowledge or hidden context."),
        ("Quality checks", "Check required sections, evidence coverage, unsupported numbers, and non-empty output."),
    ]
)
