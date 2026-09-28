"""BallDNA Streamlit navigation."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from overview import render_overview


supporting_pages = [
    st.Page("pages/2_Compare_Players.py", title="Compare Players"),
    st.Page("pages/1_Player_Scout.py", title="Scout Players"),
]
if not Path.cwd().is_relative_to("/mount/src"):
    supporting_pages.extend(
        [
            st.Page("pages/4_Ask_the_Data.py", title="Ask the Data 🚧"),
            st.Page("pages/5_Team_Needs_Lab.py", title="Team Needs Lab 🚧"),
            st.Page("pages/6_Roster_Construction_Lab.py", title="Roster Construction Lab 🚧"),
        ]
    )

navigation = st.navigation(
    {
        "": [
            st.Page(
                render_overview,
                title="Project Overview",
                icon="🏀",
                url_path="Project_Overview",
                default=True,
            ),
            st.Page(
                "pages/3_Similar_Players.py",
                title="Player Similarity",
            ),
        ],
        "Supporting tools": supporting_pages,
        "Project": [st.Page("pages/7_About_This_Project.py", title="About This Project", icon="ℹ️")],
    }
)
st.sidebar.markdown(
    """<div style="position:fixed;bottom:1rem;left:1rem;z-index:2">
    <a href="https://ko-fi.com/Q3A127SI46" target="_blank" rel="noopener noreferrer">
    <img height="36" style="border:0;height:36px"
    src="https://storage.ko-fi.com/cdn/kofi6.png?v=6" alt="Buy Me a Coffee at Ko-fi">
    </a></div>""",
    unsafe_allow_html=True,
)
navigation.run()
