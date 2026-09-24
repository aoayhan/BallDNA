"""BallDNA Streamlit navigation."""

from __future__ import annotations

import streamlit as st

from overview import render_overview


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
        "Supporting tools": [
            st.Page("pages/2_Compare_Players.py", title="Compare Players"),
            st.Page("pages/1_Player_Scout.py", title="Scout Players"),
            st.Page("pages/4_Ask_the_Data.py", title="Ask the Data 🚧"),
            st.Page("pages/5_Team_Needs_Lab.py", title="Team Needs Lab 🚧"),
            st.Page("pages/6_Roster_Construction_Lab.py", title="Roster Construction Lab 🚧"),
        ],
        "Project": [st.Page("pages/7_About_This_Project.py", title="About This Project", icon="ℹ️")],
    }
)
navigation.run()
