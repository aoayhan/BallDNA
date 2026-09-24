"""Grounded natural-language question answering page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from _bootstrap import initialize_app  # noqa: E402
from components.styles import apply_styles  # noqa: E402
from components.ui import (  # noqa: E402
    player_selector,
    render_build_trace,
    render_evidence,
    render_generation_status,
)
from ball_ai.ai.grounding import build_player_evidence  # noqa: E402
from ball_ai.ai.report_generator import answer_question  # noqa: E402
from ball_ai.config import settings  # noqa: E402
from ball_ai.data.database import (  # noqa: E402
    get_data_status_text,
    get_player_profile,
    get_players,
    get_recent_games,
)
from ball_ai.data.historical_store import get_player_season_history  # noqa: E402


st.set_page_config(page_title="Ask the Data · BallDNA", page_icon="💬", layout="wide")
apply_styles()
initialize_app()
st.title("Ask the Data")
st.warning("🚧 Under construction — this supporting workflow is not part of the current finished demo.")
st.caption(
    "Ask in natural language; grounded AI answers from a compact evidence packet and refuses unsupported causes."
)

players = get_players()
player_id = player_selector(players, "Player context", "ask_player")
if player_id is None:
    st.stop()
profile = get_player_profile(player_id)
packet = build_player_evidence(
    profile,
    get_recent_games(player_id),
    get_data_status_text(),
    season_history=get_player_season_history(player_id),
)

examples = [
    "Is this player more of a scorer or facilitator?",
    "What changed in the recent games?",
    "Does the supplied data support a spacing role?",
    "What does this player's shot selection and difficulty look like?",
    "Why is this player underperforming recently?",
]
choice = st.selectbox("Try an example", ["Write my own question", *examples])
question = st.text_area(
    "Question",
    value="" if choice == "Write my own question" else choice,
    placeholder="What does the data say about this player's recent production?",
)
prefer_llm = st.toggle("Use LLM when configured", value=bool(settings.openai_api_key))
result_key = f"qa_result_{player_id}_{hash(question.strip())}"

if st.button("Ask BallDNA", type="primary", disabled=not question.strip()):
    with st.spinner("Checking the evidence packet…"):
        st.session_state[result_key] = answer_question(packet, question, prefer_llm=prefer_llm)

if result_key in st.session_state:
    result = st.session_state[result_key]
    render_generation_status(result.mode, result.warning)
    st.markdown(result.text)

with st.expander("Inspect the evidence context", expanded=True):
    render_evidence(packet)

render_build_trace(
    [
        ("Question context", "Resolve the selected player and retrieve only the relevant stored season and recent-game rows."),
        ("Analytics layer", "Calculate compact efficiency, recent trends, and prior-season context before any prompt is created."),
        ("Grounded prompt", "Send the question, approved evidence IDs, provenance, and limitations—not the raw database—to the model."),
        ("Safe response", "Require cited claims and an explicit limitation; deterministic mode refuses questions the supplied data cannot answer."),
    ]
)
