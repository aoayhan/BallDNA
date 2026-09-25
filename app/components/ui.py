"""UI helpers shared across Streamlit pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler

from ball_ai.ai.grounding import evidence_frame
from ball_ai.analytics.tables import basketball_reference_season_table
from ball_ai.data.player_filters import resolve_player_search


TEAM_NAMES = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


def _table_float(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _team_label(team: str) -> str:
    return f"{TEAM_NAMES.get(team, team)} · {team}"


def render_copyable_table(frame: pd.DataFrame | Styler) -> None:
    """Render a horizontally scrollable table whose text remains selectable."""

    if isinstance(frame, Styler):
        html = frame.hide(axis="index").to_html()
    else:
        html = frame.to_html(
            index=False,
            border=0,
            escape=True,
            na_rep="—",
            float_format=_table_float,
        )
    st.html(f'<div class="copyable-table">{html}</div>')


def _player_option_label(row: object) -> str:
    display_team = getattr(row, "display_team", getattr(row, "team", None))
    if pd.isna(display_team) or not str(display_team).strip():
        return f"{row.player_name} · {row.position}"
    return f"{row.player_name} · {display_team} · {row.position}"


def player_selector(
    players: pd.DataFrame,
    label: str,
    key: str,
    *,
    default_player_name: str | None = None,
    compact: bool = False,
    show_count: bool = True,
    single_box: bool = False,
) -> int | None:
    """Render explicit team and name filters followed by a short player list."""

    if single_box:
        labels = {
            int(row.player_id): _player_option_label(row)
            for row in players.itertuples()
        }
        default_id = next(
            (
                int(row.player_id)
                for row in players.itertuples()
                if str(row.player_name).casefold() == str(default_player_name).casefold()
            ),
            next(iter(labels)),
        )
        selected = st.selectbox(
            label,
            list(labels),
            index=list(labels).index(default_id),
            format_func=labels.get,
            key=f"{key}_player",
        )
        return int(selected)

    st.markdown(f"**{label}**")
    teams = sorted(players["team"].dropna().astype(str).unique())
    columns = st.columns([.85, 1.15, 1.35] if compact else [1, 1.6])
    filter_column, search_column = columns[:2]
    with filter_column:
        team = st.selectbox(
            f"{label} team",
            [None, *teams],
            format_func=lambda value: "All teams" if value is None else _team_label(value),
            key=f"{key}_team",
        )
    with search_column:
        query = st.text_input(
            f"Search {label}",
            placeholder="Type a player name…",
            key=f"{key}_search",
        )

    search_result = resolve_player_search(players, team=team, query=query)
    matches = search_result.matches
    if matches.empty:
        st.warning("No players match these filters. Try another team or search term.")
        return None

    if search_result.used_league_wide_fallback:
        selected_team = _team_label(str(team))
        st.info(
            f'No player matching “{query.strip()}” was found on **{selected_team}**. '
            "Here are name matches from other teams:"
        )
        recommendation_rows = []
        for row in matches.head(5).itertuples():
            display_team = getattr(row, "display_team", row.team)
            recommendation_rows.append(
                f"- {row.player_name} · **{_team_label(str(display_team))}** · {row.position}"
                if pd.notna(display_team) and str(display_team).strip()
                else f"- {row.player_name} · {row.position}"
            )
        recommendations = "\n".join(recommendation_rows)
        if len(matches) > 5:
            recommendations += f"\n- …and {len(matches) - 5} more matching players"
        st.markdown(recommendations)

    labels = {
        int(row.player_id): _player_option_label(row)
        for row in matches.itertuples()
    }
    default_index = next(
        (
            index
            for index, row in enumerate(matches.itertuples())
            if str(row.player_name).casefold()
            == str(default_player_name).casefold()
        ),
        0,
    )
    suffix = (
        " recommended outside the selected team"
        if search_result.used_league_wide_fallback
        else " available"
    )
    player_column = columns[2] if compact else st.container()
    with player_column:
        selected = st.selectbox(
            f"Choose {label}",
            list(labels),
            index=default_index,
            format_func=labels.get,
            key=f"{key}_player",
        )
        if show_count:
            st.caption(f"{len(matches)} player{'s' if len(matches) != 1 else ''}{suffix}")
    return int(selected)


def render_profile_metrics(profile: dict) -> None:
    """Render one horizontal Basketball Reference-style season row."""

    render_copyable_table(basketball_reference_season_table([profile]))


def render_evidence(packet: dict) -> None:
    frame = evidence_frame(packet).rename(
        columns={
            "evidence_id": "ID",
            "label": "Evidence",
            "display_value": "Value",
            "context": "Context",
        }
    )
    render_copyable_table(frame)


def render_build_trace(
    steps: list[tuple[str, str]],
    *,
    title: str = "How this was built",
    expanded: bool = False,
) -> None:
    """Expose an AI/analytics execution trace in recruiter-readable language."""

    with st.expander(title, expanded=expanded):
        for index, (name, detail) in enumerate(steps, start=1):
            st.markdown(f"**{index}. {name}**  \n{detail}")
