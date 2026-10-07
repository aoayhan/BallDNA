"""Short landing page for Basketball DNA."""

from base64 import b64encode
from pathlib import Path

import streamlit as st

from _bootstrap import initialize_app
from components.styles import apply_styles


def render_overview() -> None:
    """Introduce Player DNA and send visitors to the flagship workflow."""

    st.set_page_config(page_title="Basketball DNA", page_icon="🏀", layout="wide")
    apply_styles()
    initialize_app()
    iverson_data = b64encode(
        (Path(__file__).parent / "static" / "allen_iverson_cc_by_sa.jpg").read_bytes()
    ).decode("ascii")
    st.markdown(
        f"""
        <style>
        .overview-hero {{
            position: relative;
            overflow: hidden;
            min-height: 390px;
            padding: 3.4rem 2.7rem;
            border: 1px solid rgba(255,107,53,.28);
            border-radius: 24px;
            background: #f8fbff;
        }}
        .overview-hero::after {{
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(90deg, #f8fbff 0%, #f8fbff 48%, rgba(248,251,255,.76) 64%, rgba(248,251,255,.1) 100%),
                url("data:image/jpeg;base64,{iverson_data}");
            background-position: center, 92% 40%;
            background-size: cover, 52% auto;
            background-repeat: no-repeat;
            filter: grayscale(1);
        }}
        .overview-hero > * {{position: relative; z-index: 1;}}
        @media (max-width: 720px) {{
            .overview-hero {{padding: 2rem 1.5rem;}}
            .overview-hero h1 {{font-size: 2.5rem !important;}}
            .overview-hero::after {{opacity: .55;}}
        }}
        .overview-hero h1 {{font-size: 3.2rem; line-height: 1.02; margin: .5rem 0 1rem; max-width: 650px;}}
        .overview-hero p {{color: #34435a; max-width: 610px; font-size: 1.08rem;}}
        .overview-joke {{color: #69778b !important; font-size: .9rem !important; margin-top: 1.2rem;}}
        .st-key-overview_similarity_button button {{
            width: 100%;
            color: white;
            font-weight: 800;
            border-color: #b63d12;
            background:
                radial-gradient(ellipse at -12% 50%, transparent 56%, rgba(74,24,7,.25) 57% 59%, transparent 60%),
                radial-gradient(ellipse at 112% 50%, transparent 56%, rgba(74,24,7,.25) 57% 59%, transparent 60%),
                linear-gradient(90deg, transparent 49%, rgba(74,24,7,.22) 50%, transparent 51%),
                #f36b2b;
        }}
        .st-key-overview_similarity_button button:hover {{
            color: white; border-color: #8f2f0d; background-color: #dc541d;
        }}
        </style>
        <section class="overview-hero">
            <h1>Find out who your favorite player plays like.</h1>
            <p>Basketball DNA learns Player DNA from NBA shot selection, creation, playmaking,
            and defensive actions, then searches basketball history for the closest style matches.</p>
            <p class="overview-joke">AI here means embeddings—not Allen Iverson, although he is in the search pool.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    _, cta, _ = st.columns([1, .4, 1])
    with cta:
        if st.button(
            "Explore Similarity",
            type="primary",
            key="overview_similarity_button",
            width="stretch",
        ):
            st.switch_page("pages/3_Similar_Players.py")

    first, second, third = st.columns(3)
    first.markdown("**Choose a player-season**  \nCompare offensive, defensive, or overall behavior across modern NBA history.")
    second.markdown("**Retrieve style matches**  \nUse self-supervised Player DNA, not names, teams, or positions.")
    third.markdown("**Inspect the evidence**  \nSee shared tendencies, largest differences, reliability, and separate impact context.")
