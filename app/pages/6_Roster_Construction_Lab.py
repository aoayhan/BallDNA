"""Interactive counterfactual roster construction over learned team profiles."""

from __future__ import annotations

import json
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
    TEAM_NAMES,
    render_build_trace,
    render_copyable_table,
)
from ball_ai.ai.grounding import build_roster_simulation_evidence  # noqa: E402
from ball_ai.ai.report_generator import generate_roster_simulation_report  # noqa: E402
from ball_ai.analytics.roster_construction import (  # noqa: E402
    MODEL_FEATURES,
    allocate_reference_rotation,
    assess_embedding_support,
    anchor_prediction_to_reference,
    assemble_simulated_roster,
    embedding_changes,
    fit_roster_quality_model,
    fit_talent_ceiling_model,
    fit_win_calibrator,
    nearest_historical_teams,
    predict_roster_quality,
    predict_talent_ceiling,
    primary_team_player_pool,
    profile_scores,
    roster_embedding,
    roster_talent_summary,
)
from ball_ai.config import settings  # noqa: E402
from ball_ai.data.player_filters import (  # noqa: E402
    order_player_candidates,
    strict_player_name_matches,
)


st.set_page_config(
    page_title="Roster Construction Lab · BallDNA",
    page_icon="🧬",
    layout="wide",
)
apply_styles()
initialize_app()
@st.cache_data(show_spinner=False)
def load_roster_assets() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    root = settings.historical_data_dir
    player_path = root / "roster_player_team_vectors.parquet"
    team_path = root / "roster_team_embeddings.parquet"
    evaluation_path = root / "roster_model_evaluation.json"
    if not all(path.exists() for path in (player_path, team_path, evaluation_path)):
        return pd.DataFrame(), pd.DataFrame(), {}
    return (
        pd.read_parquet(player_path),
        pd.read_parquet(team_path),
        json.loads(evaluation_path.read_text(encoding="utf-8")),
    )


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    _, dataset, evaluation = load_roster_assets()
    model = fit_roster_quality_model(dataset, evaluation["selected_model"])
    ceiling_model = fit_talent_ceiling_model(dataset)
    calibrator = fit_win_calibrator(dataset)
    return model, ceiling_model, calibrator


def team_label(abbreviation: str) -> str:
    return f"{TEAM_NAMES.get(abbreviation, abbreviation)} · {abbreviation}"


def ordinal(value: float) -> str:
    """Format a rounded percentile with the correct English suffix."""

    number = int(round(value))
    suffix = (
        "th"
        if 10 <= number % 100 <= 20
        else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    )
    return f"{number}{suffix}"


def queue_player(
    state_key: str,
    selector_key: str,
    candidates: pd.DataFrame,
) -> None:
    """Queue an exact option or resolve a submitted near-spelling."""

    selected_value = st.session_state.get(selector_key)
    if selected_value is None:
        return
    if isinstance(selected_value, str):
        matches = strict_player_name_matches(candidates, selected_value)
        if matches.empty:
            st.session_state[f"{selector_key}_message"] = (
                f'No close player-name match for "{selected_value}".'
            )
            st.session_state[selector_key] = None
            return
        selected_id = int(matches.iloc[0]["player_id"])
    else:
        selected_id = int(selected_value)
    current = list(st.session_state.get(state_key, []))
    if selected_id not in current:
        current.append(selected_id)
    st.session_state[state_key] = current
    st.session_state[selector_key] = None
    st.session_state.pop(f"{selector_key}_message", None)


def add_player_control(pool: pd.DataFrame, state_key: str) -> None:
    st.markdown("**Add players**")
    left, right = st.columns([1, 1.8])
    teams = sorted(pool["team"].dropna().unique())
    with left:
        source_team = st.selectbox(
            "Source team",
            teams,
            index=None,
            format_func=team_label,
            key=f"{state_key}_source_team",
            placeholder="All teams",
        )
    candidates = order_player_candidates(
        pool,
        preferred_team=source_team,
        excluded_ids=st.session_state.get(state_key, []),
    )
    labels = {
        int(row.player_id): (
            f"{row.player_name} · {TEAM_NAMES.get(str(row.team), str(row.team))} "
            f"({row.team}) · {row.position}"
        )
        for row in candidates.itertuples()
    }
    selector_key = f"{state_key}_candidate"
    with right:
        st.selectbox(
            "Search player to add",
            list(labels),
            index=None,
            format_func=labels.get,
            key=selector_key,
            placeholder="Type a player name, then press Enter to add…",
            filter_mode="contains",
            accept_new_options=True,
            on_change=queue_player,
            args=(state_key, selector_key, candidates),
        )
        if message := st.session_state.get(f"{selector_key}_message"):
            st.caption(message)
    if source_team:
        st.caption(
            f"Showing only **{team_label(str(source_team))}** players. Use the × in the team "
            "box to return to the full league."
        )
    else:
        st.caption(
            "Dropdown results use literal name matching. For a one- or two-letter typo, "
            "type the name and press Enter to select the closest valid player."
        )


st.title("Roster Construction Lab")
st.warning("🚧 Under construction — treat roster simulations as experimental, not validated forecasts.")
st.caption(
    "Build a counterfactual NBA roster in player-embedding space. A validated ML model "
    "estimates the team profile; deterministic summaries report fixed results without changing them."
)
st.info(
    "DARKO DPM supplies the player-impact estimate; BallDNA's per-36 and shot-profile traits "
    "describe style and fit. Minutes are not player quality. Actual season minute shares represent "
    "role and availability when players are pooled into a team."
)

player_stints, team_dataset, evaluation = load_roster_assets()
if player_stints.empty or team_dataset.empty or not evaluation:
    st.error(
        "Roster assets are missing. Run `python scripts/build_roster_construction_assets.py`."
    )
    st.stop()

model, ceiling_model, win_calibrator = load_model_bundle()
seasons = list(reversed(evaluation["seasons"]))
season_column, team_column, mode_column = st.columns([1, 1.4, 1.2])
with season_column:
    season = st.selectbox("Season", seasons)
season_teams = team_dataset.loc[team_dataset["season"].eq(season)].sort_values("league_rank")
team_options = {
    f"#{int(row.league_rank)} · {team_label(row.team_abbreviation)} · "
    f"{int(row.wins)}-{int(row.games-row.wins)}": str(row.team_abbreviation)
    for row in season_teams.itertuples()
}
with team_column:
    selected_team_label = st.selectbox("Reference team", list(team_options))
with mode_column:
    start_mode = st.radio(
        "Starting roster",
        ["Reference team", "Empty custom roster"],
        horizontal=True,
    )
team = team_options[selected_team_label]
reference_row = season_teams.loc[season_teams["team_abbreviation"].eq(team)].iloc[0]
pool = primary_team_player_pool(player_stints, season)
candidate_pool = pool.loc[pool["games_played"].ge(5)].copy()
reference_pool = pool.loc[pool["team"].eq(team)].copy()
reference_ids = reference_pool["player_id"].astype(int).tolist()
candidate_pool = candidate_pool.loc[
    ~candidate_pool["player_id"].astype(int).isin(reference_ids)
]
if len(reference_ids) < 5:
    st.error("The selected reference team does not have enough assigned players in this snapshot.")
    st.stop()

reference_rotation = allocate_reference_rotation(reference_pool, pool, reference_ids)
baseline_embedding = roster_embedding(
    reference_rotation, minute_column="normalized_minutes"
)
error = float(evaluation["error_interval"]["net_rating_points"])
baseline_prediction = predict_roster_quality(
    baseline_embedding, model, win_calibrator, net_rating_error=error
)

state_key = f"roster_additions_{season}_{team}_{start_mode}"
if state_key not in st.session_state:
    st.session_state[state_key] = []
eligible_addition_ids = set(candidate_pool["player_id"].astype(int))
st.session_state[state_key] = [
    player_id
    for player_id in st.session_state[state_key]
    if int(player_id) in eligible_addition_ids
]
forced_key = f"{state_key}_forced_rotation"
if forced_key not in st.session_state:
    st.session_state[forced_key] = []
removal_key = f"roster_removals_{season}_{team}_{start_mode}"
if removal_key not in st.session_state:
    st.session_state[removal_key] = []
base_ids = reference_ids if start_mode == "Reference team" else []

st.divider()
builder_left, builder_right = st.columns([1, 1.05])
with builder_left:
    st.subheader("1 · Edit the roster")
    if start_mode != "Reference team":
        st.info("Start from zero and add at least five players from the league-wide pool.")
    add_player_control(candidate_pool, state_key)
    st.caption("Addition eligibility: at least five games played in the selected season.")
    added_ids = list(st.session_state[state_key])
    if added_ids:
        added_rows = (
            pool.loc[pool["player_id"].isin(added_ids)]
            .assign(
                queue_order=lambda frame: frame["player_id"].astype(int).map(
                    {player_id: index for index, player_id in enumerate(added_ids)}
                )
            )
            .sort_values("queue_order")
        )
        st.markdown("**Queued additions**")
        for row in added_rows.itertuples():
            player_column, remove_column = st.columns([5, 1])
            player_column.markdown(
                f"{row.player_name} · **{team_label(str(row.team))}** · {row.position}"
            )
            if remove_column.button(
                "Remove",
                key=f"{state_key}_remove_{int(row.player_id)}",
                width="stretch",
            ):
                st.session_state[state_key] = [
                    player_id
                    for player_id in added_ids
                    if player_id != int(row.player_id)
                ]
                st.session_state[forced_key] = [
                    player_id
                    for player_id in st.session_state[forced_key]
                    if player_id != int(row.player_id)
                ]
                st.rerun()
        if st.button("Clear all additions", key=f"{state_key}_clear"):
            st.session_state[state_key] = []
            st.session_state[forced_key] = []
            st.rerun()

    if start_mode == "Reference team":
        st.divider()
        st.markdown("**Remove players**")
        removed_ids = list(st.session_state[removal_key])
        removal_candidates = reference_pool.loc[
            ~reference_pool["player_id"].astype(int).isin(removed_ids)
        ].sort_values("team_minutes", ascending=False)
        remove_labels = {
            int(row.player_id): (
                f"{row.player_name} · {row.position} · "
                f"{row.minutes_per_game:.1f} observed MPG"
            )
            for row in removal_candidates.itertuples()
        }
        removal_selector_key = f"{removal_key}_candidate"
        st.selectbox(
            "Remove reference-team players",
            list(remove_labels),
            index=None,
            format_func=remove_labels.get,
            key=removal_selector_key,
            placeholder="Type a player name, then press Enter to remove…",
            filter_mode="contains",
            on_change=queue_player,
            args=(removal_key, removal_selector_key, removal_candidates),
        )
        if message := st.session_state.get(f"{removal_selector_key}_message"):
            st.caption(message)
        if removed_ids:
            st.markdown("**Queued removals**")
            removed_rows = (
                reference_pool.loc[reference_pool["player_id"].isin(removed_ids)]
                .assign(
                    queue_order=lambda frame: frame["player_id"].astype(int).map(
                        {
                            player_id: index
                            for index, player_id in enumerate(removed_ids)
                        }
                    )
                )
                .sort_values("queue_order")
            )
            for row in removed_rows.itertuples():
                player_column, restore_column = st.columns([5, 1])
                player_column.markdown(
                    f"{row.player_name} · **{team_label(team)}** · {row.position}"
                )
                if restore_column.button(
                    "Restore",
                    key=f"{removal_key}_restore_{int(row.player_id)}",
                    width="stretch",
                ):
                    st.session_state[removal_key] = [
                        player_id
                        for player_id in removed_ids
                        if player_id != int(row.player_id)
                    ]
                    st.rerun()
            if st.button("Clear all removals", key=f"{removal_key}_clear"):
                st.session_state[removal_key] = []
                st.rerun()
    else:
        removed_ids = []

final_ids = [player_id for player_id in base_ids if player_id not in removed_ids]
final_ids.extend(player_id for player_id in added_ids if player_id not in final_ids)
if len(final_ids) < 5:
    with builder_right:
        st.subheader("2 · Review automatic role weights")
        st.warning(f"Add at least {5-len(final_ids)} more player(s) to create a valid roster.")
    st.stop()

if start_mode == "Reference team":
    forced_ids = [
        player_id
        for player_id in st.session_state[forced_key]
        if player_id in added_ids
    ]
    initial_roster = allocate_reference_rotation(
        reference_pool,
        pool,
        final_ids,
        forced_player_ids=forced_ids,
    )
else:
    forced_ids = []
    initial_roster = assemble_simulated_roster(pool, final_ids)
with builder_right:
    st.subheader("2 · Review automatic role weights")
    st.caption(
        "Every incumbent keeps a non-zero share based on the total minutes he actually contributed. "
        "The 240-minute equivalent makes those season shares readable; it is not a claim that every "
        "player appeared in the same game. An addition can earn up "
        "to that ceiling by taking minutes from lower-impact players at overlapping positions; open "
        "minutes from removals are filled first."
    )
    baseline_rotation = (
        reference_rotation.set_index("player_id")["normalized_minutes"]
        if start_mode == "Reference team"
        else pd.Series(dtype=float)
    )
    rotation_display = initial_roster[
        [
            "player_name", "team", "position", "games_played",
            "minutes_per_game", "normalized_minutes", "model_rotation_share",
            "impact_rating", "offensive_impact", "defensive_impact", "impact_source",
            "role_strength", "data_reliability",
        ]
    ].copy()
    rotation_display["rotation_change"] = initial_roster["player_id"].map(
        baseline_rotation
    ).fillna(0).rsub(initial_roster["normalized_minutes"])
    rotation_display = rotation_display.rename(
        columns={
            "player_name": "Player",
            "team": "Source team",
            "position": "Pos",
            "games_played": "G",
            "minutes_per_game": "Observed MPG",
            "normalized_minutes": "240-minute role equivalent",
            "rotation_change": "Δ role equivalent",
            "model_rotation_share": "Model role weight",
            "impact_rating": "DPM",
            "offensive_impact": "O-DPM",
            "defensive_impact": "D-DPM",
            "impact_source": "Impact source",
            "role_strength": "Role context",
            "data_reliability": "Data reliability",
        }
    )
    rotation_display["Observed MPG"] = rotation_display["Observed MPG"].map(
        lambda value: f"{value:.1f}"
    )
    rotation_display["240-minute role equivalent"] = rotation_display[
        "240-minute role equivalent"
    ].map(lambda value: f"{value:.1f}")
    rotation_display["Δ role equivalent"] = rotation_display["Δ role equivalent"].map(
        lambda value: f"{value:+.1f}" if start_mode == "Reference team" else "—"
    )
    rotation_display["Model role weight"] = rotation_display["Model role weight"].map(
        lambda value: f"{value:.1%}"
    )
    for column in ("DPM", "O-DPM", "D-DPM"):
        rotation_display[column] = rotation_display[column].map(
            lambda value: "—" if pd.isna(value) else f"{value:+.1f}"
        )
    rotation_display["Role context"] = rotation_display["Role context"].map(
        lambda value: f"{value:.0%}ile"
    )
    rotation_display["Data reliability"] = rotation_display["Data reliability"].map(
        lambda value: f"{value:.0%}"
    )
    render_copyable_table(rotation_display)
    if added_ids:
        added_rotation = initial_roster.loc[
            initial_roster["player_id"].astype(int).isin(added_ids),
            ["player_name", "minutes_per_game", "normalized_minutes"],
        ].copy()
        added_rotation["unused_source_minutes"] = (
            added_rotation["minutes_per_game"] - added_rotation["normalized_minutes"]
        ).clip(lower=0)
        added_rotation = added_rotation.rename(
            columns={
                "player_name": "Addition",
                "minutes_per_game": "MPG on source team",
                "normalized_minutes": "MPG earned here",
                "unused_source_minutes": "MPG compressed by competition",
            }
        )
        st.markdown("**How much of each addition actually enters the rotation**")
        render_copyable_table(
            added_rotation.style.format(
                {
                    "MPG on source team": "{:.1f}",
                    "MPG earned here": "{:.1f}",
                    "MPG compressed by competition": "{:.1f}",
                }
            ),
        )
        st.caption(
            "A player who cannot beat an incumbent for overlapping minutes receives little or no "
            "modeled role—and therefore should not create several wins merely by being added."
        )
        if start_mode == "Reference team":
            audit_by_id = {
                int(item["player_id"]): item
                for item in initial_roster.attrs.get("allocation_audit", [])
            }
            for row in added_rows.itertuples():
                player_id = int(row.player_id)
                audit = audit_by_id.get(player_id)
                if not audit:
                    continue
                assigned = float(audit["assigned_minutes"])
                requested = float(audit["requested_minutes"])
                displaced = audit["displacements"]
                blockers = audit["blocked_by"][:4]
                with st.container(border=True):
                    st.markdown(
                        f"**ⓘ Why {row.player_name} receives {assigned:.1f} MPG**"
                    )
                    if displaced:
                        displacement_text = ", ".join(
                            f"**{item['player_name']}** ({item['minutes']:.1f})"
                            for item in displaced
                        )
                        st.write(
                            f"The allocator takes minutes from {displacement_text}."
                        )
                    elif float(audit["open_minutes_used"]) > 0:
                        st.write(
                            f"The player fills {audit['open_minutes_used']:.1f} minutes opened by removals."
                        )
                    else:
                        st.write("The player does not displace anyone automatically.")

                    if assigned + 0.05 < requested and blockers:
                        blocker_text = ", ".join(
                            f"**{item['player_name']}** "
                            f"({item['impact_rating']:+.1f} DPM)"
                            if pd.notna(item.get("impact_rating"))
                            else f"**{item['player_name']}** (fallback score)"
                            for item in blockers
                        )
                        addition_score = (
                            f"{audit['impact_rating']:+.1f} DPM"
                            if pd.notna(audit.get("impact_rating"))
                            else "the fallback impact score"
                        )
                        st.write(
                            f"The model gives {row.player_name} a lower impact rating "
                            f"({addition_score}) than overlapping options such as "
                            f"{blocker_text}. It therefore leaves {requested-assigned:.1f} of the "
                            "player's source-team MPG outside this rotation."
                        )
                        if assigned < 0.5:
                            st.info(
                                "Because the player earns no meaningful automatic role, this "
                                "addition does not add expected wins. Use the override below to "
                                "make the player take minutes from those higher-rated options."
                            )

                    if audit["forced"]:
                        st.warning(
                            "Forced-role experiment is active. These minutes are a user assumption, "
                            "not the allocator's recommendation."
                        )
                        if st.button(
                            "Return to merit-based role",
                            key=f"{state_key}_unforce_{player_id}",
                            width="stretch",
                        ):
                            st.session_state[forced_key] = [
                                value
                                for value in st.session_state[forced_key]
                                if value != player_id
                            ]
                            st.rerun()
                    elif assigned + 0.05 < requested:
                        if st.button(
                            f"Force {row.player_name} into the rotation",
                            key=f"{state_key}_force_{player_id}",
                            help=(
                                "Give this player up to the MPG recorded on the source team, taking "
                                "minutes from the lowest-impact overlapping players even when they "
                                "rate higher."
                            ),
                            width="stretch",
                        ):
                            st.session_state[forced_key] = [
                                *st.session_state[forced_key],
                                player_id,
                            ]
                            st.rerun()

simulated_roster = initial_roster
simulated_embedding = roster_embedding(
    simulated_roster, minute_column="normalized_minutes"
)
simulated_prediction = predict_roster_quality(
    simulated_embedding, model, win_calibrator, net_rating_error=error
)
baseline_ceiling = predict_talent_ceiling(
    baseline_embedding, ceiling_model, win_calibrator
)
simulated_ceiling = predict_talent_ceiling(
    simulated_embedding, ceiling_model, win_calibrator
)
support = assess_embedding_support(simulated_embedding, team_dataset)
talent = roster_talent_summary(simulated_embedding, team_dataset)
quality_supported = bool(support["is_supported"])
if start_mode == "Reference team":
    displayed_baseline_prediction = {
        **baseline_prediction,
        "net_rating": float(reference_row["net_rating"]),
        "expected_wins": 82 * float(reference_row["wins"]) / float(reference_row["games"]),
    }
    displayed_prediction = anchor_prediction_to_reference(
        simulated_prediction,
        baseline_prediction,
        observed_net_rating=float(reference_row["net_rating"]),
        observed_wins=float(reference_row["wins"]),
        observed_games=float(reference_row["games"]),
        win_calibrator=win_calibrator,
        net_rating_error=error,
    )
    ceiling_net_rating = float(reference_row["net_rating"]) + float(
        simulated_ceiling["net_rating"] - baseline_ceiling["net_rating"]
    )
    observed_win_pace = 82 * float(reference_row["wins"]) / float(reference_row["games"])
    ceiling_wins = observed_win_pace + float(
        win_calibrator.predict(
            pd.DataFrame({"net_rating": [ceiling_net_rating]})
        )[0]
        - win_calibrator.predict(
            pd.DataFrame({"net_rating": [float(reference_row["net_rating"])]})
        )[0]
    )
    displayed_ceiling = {
        "net_rating": ceiling_net_rating,
        "expected_wins": max(0.0, min(82.0, ceiling_wins)),
        "display_wins": int(round(max(0.0, min(82.0, ceiling_wins)))),
    }
else:
    displayed_baseline_prediction = baseline_prediction
    displayed_prediction = simulated_prediction
    displayed_ceiling = simulated_ceiling
reference_actual_win_pace = 82 * float(reference_row["wins"]) / float(
    reference_row["games"]
)
reference_net_rating_implied_wins = float(
    win_calibrator.predict(
        pd.DataFrame({"net_rating": [float(reference_row["net_rating"])]})
    )[0]
)
reference_wins_vs_implied = (
    reference_actual_win_pace - reference_net_rating_implied_wins
)
changes = embedding_changes(baseline_embedding, simulated_embedding)
baseline_profiles = profile_scores(
    baseline_embedding,
    reference_rotation,
    minute_column="normalized_minutes",
    comparison_pool=pool,
).rename(
    columns={"score": "Baseline"}
)
simulated_profiles = profile_scores(
    simulated_embedding,
    simulated_roster,
    minute_column="normalized_minutes",
    comparison_pool=pool,
).rename(
    columns={"score": "Simulated"}
)
profile_comparison = baseline_profiles.merge(simulated_profiles, on="profile")
simulated_profiles = simulated_profiles.rename(columns={"Simulated": "score"})
neighbours = nearest_historical_teams(
    simulated_embedding,
    team_dataset,
    top_k=6,
    exclude=(season, int(reference_row["team_id"])),
)

removed_names = (
    reference_pool.loc[reference_pool["player_id"].isin(removed_ids), "player_name"]
    .astype(str)
    .tolist()
)
added_names = (
    pool.loc[pool["player_id"].isin(added_ids), "player_name"].astype(str).tolist()
)
packet = build_roster_simulation_evidence(
    season=season,
    baseline_team={
        **reference_row[
            [
                "team", "team_abbreviation", "season", "wins", "games",
                "net_rating", "league_rank",
            ]
        ].to_dict(),
        "actual_82_game_win_pace": reference_actual_win_pace,
        "net_rating_implied_wins": reference_net_rating_implied_wins,
        "actual_minus_implied_wins": reference_wins_vs_implied,
    },
    baseline_prediction=displayed_baseline_prediction,
    simulated_prediction=displayed_prediction,
    roster=simulated_roster,
    added_players=added_names,
    removed_players=removed_names,
    changes=changes,
    profiles=simulated_profiles,
    neighbours=neighbours,
    model_evaluation=evaluation,
    support=support,
    talent=talent,
    ceiling=displayed_ceiling,
    forced_players=(
        pool.loc[pool["player_id"].isin(forced_ids), "player_name"].astype(str).tolist()
    ),
)

st.divider()
st.subheader("3 · Inspect the model output")
delta = displayed_prediction["net_rating"] - displayed_baseline_prediction["net_rating"]
model_resolution = float(
    evaluation.get("minimum_detectable_change", evaluation["selected_model_mae"])
)
detectable_change = quality_supported and abs(delta) >= model_resolution
is_counterfactual = bool(added_ids or removed_ids) or start_mode == "Empty custom roster"
st.markdown(
    "**Experimental counterfactual outlook**"
    if is_counterfactual
    else "**Observed team baseline**"
)
upside_columns = st.columns(4)
ceiling_wins = int(displayed_ceiling["display_wins"])
if is_counterfactual:
    upside_columns[0].metric(
        "Experimental expected wins",
        str(ceiling_wins),
        f"{ceiling_wins}-{82-ceiling_wins} projected record",
    )
    upside_columns[1].metric(
        "Experimental Net Rating", f"{displayed_ceiling['net_rating']:+.1f}"
    )
else:
    upside_columns[0].metric(
        "Actual wins",
        str(int(reference_row["wins"])),
        f"{int(reference_row['wins'])}-{int(reference_row['games']-reference_row['wins'])} record",
    )
    upside_columns[1].metric(
        "Actual Net Rating", f"{float(reference_row['net_rating']):+.1f}"
    )
upside_columns[2].metric(
    "Historical talent percentile", ordinal(talent["historical_percentile"])
)
active_rotation = int(simulated_roster["normalized_minutes"].gt(4.0).sum())
upside_columns[3].metric("Active rotation (>4 MPG)", str(active_rotation))
with st.container(border=True):
    st.markdown("**Reference result vs Net Rating expectation**")
    result_columns = st.columns(3)
    actual_label = (
        "Actual wins"
        if int(reference_row["games"]) == 82
        else "Actual 82-game win pace"
    )
    result_columns[0].metric(actual_label, f"{reference_actual_win_pace:.1f}")
    result_columns[1].metric(
        "Net Rating-implied wins", f"{reference_net_rating_implied_wins:.1f}"
    )
    result_columns[2].metric(
        "Actual − implied", f"{reference_wins_vs_implied:+.1f}"
    )
    if abs(reference_wins_vs_implied) < 0.05:
        result_read = "finished exactly in line with"
    elif reference_wins_vs_implied > 0:
        result_read = "finished above"
    else:
        result_read = "finished below"
    st.caption(
        f"Based only on its actual {float(reference_row['net_rating']):+.1f} Net Rating, "
        f"the historical calibration implies {reference_net_rating_implied_wins:.1f} wins. "
        f"The team {result_read} that expectation by {abs(reference_wins_vs_implied):.1f} wins. "
        "This is a retrospective result diagnostic, not a preseason prediction."
    )
if is_counterfactual:
    if forced_ids:
        st.info(
            "A forced-role experiment is active, so the outlook can fall when a lower-impact player "
            "takes minutes from stronger overlapping options. The minutes are your scenario assumption, "
            "not a coaching prediction; historical transaction outcomes have not calibrated this as a "
            "real forecast."
        )
    else:
        st.info(
            "The merit-based outlook is intentionally monotonic: an addition cannot make the team worse "
            "when the allocator can leave that player outside the rotation. Expected wins is an "
            "experimental comparison point; historical transaction outcomes have not yet calibrated it "
            "as a real forecast."
        )
    if not quality_supported:
        st.warning(
            f"The roster is {support['distance_ratio']:.1f}× beyond the empirical historical "
            "support boundary. The identity and talent-upside score remain available, but the "
            "cross-sectional association is hidden."
        )
    with st.expander("Historical profile association · diagnostic only"):
        if quality_supported:
            st.write(
                f"The same-season association model maps this profile to "
                f"**{displayed_prediction['net_rating']:+.1f} Net Rating** and a "
                f"**{displayed_prediction['expected_wins']:.0f}-win pace**. Its change versus the "
                f"reference is {delta:+.2f}, but this model is validated on observed teams—not on "
                "player additions—so the sign is not treated as a trade forecast."
            )
        else:
            st.write(
                "The association model is outside its historical support boundary, so its raw result is hidden."
            )
        st.write(
            f"Support: **{'inside' if quality_supported else 'outside'} historical range** · "
            f"leave-one-season-out team-profile MAE: **{evaluation['selected_model_mae']:.2f} Net Rating**."
        )
elif quality_supported:
    st.caption(
        f"Reference team actual: {reference_row['net_rating']:+.1f} Net Rating and "
        f"{int(reference_row['wins'])}-{int(reference_row['games']-reference_row['wins'])}. "
        "The model reproduces the unchanged baseline before any roster edit."
    )
else:
    outside_count = len(support["features_outside_training_range"])
    st.error(
        "This roster is outside the historical support of the quality model, so BallDNA will not "
        "present its raw regression result as a Net Rating or win forecast. Its nearest historical "
        f"roster is {support['distance_ratio']:.1f}× the empirical support boundary away, and "
        f"{outside_count} model feature(s) exceed every training team. The identity chart remains "
        "valid as a descriptive embedding."
    )
    st.caption(
        f"The roster's descriptive talent index is {talent['talent_index']:.3f}, at the "
        f"{talent['historical_percentile']:.0f}th percentile of {len(team_dataset)} historical "
        "team-seasons. This is not converted into wins because lineup roles, usage redistribution, "
        "chemistry, health, and diminishing returns are not modeled."
    )
profile_tab, change_tab, neighbour_tab, ai_tab, evidence_tab = st.tabs(
    ["Team identity", "Embedding changes", "Historical neighbours", "Rule-based summary", "Model evidence"]
)
with profile_tab:
    profile_long = profile_comparison.melt(
        id_vars="profile", var_name="Roster", value_name="score"
    )
    chart = px.bar(
        profile_long,
        x="score",
        y="profile",
        color="Roster",
        barmode="group",
        orientation="h",
        range_x=[0, 1],
        labels={"score": "Season-relative profile score", "profile": ""},
        color_discrete_map={"Baseline": "#64748b", "Simulated": "#ff6b35"},
    )
    chart.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(chart, width="stretch")
    defensive_rank = int(
        season_teams["defensive_rating"].rank(method="min").loc[reference_row.name]
    )
    st.caption(
        f"Reference team actual defensive rating: {float(reference_row['defensive_rating']):.1f} "
        f"(#{defensive_rank} of {len(season_teams)}, lower is better). “Interior defensive "
        "tools” is a frontcourt-relative blocks and defensive-rebounds profile—not a team "
        "defense grade."
    )
    strongest = simulated_profiles.head(3)
    weakest = simulated_profiles.tail(2).sort_values("score")
    identity_left, identity_right = st.columns(2)
    with identity_left:
        st.markdown("**Strongest profile dimensions**")
        for row in strongest.itertuples():
            st.write(f"{row.profile}: {row.score:.0%}")
    with identity_right:
        st.markdown("**Potential remaining needs**")
        for row in weakest.itertuples():
            st.write(f"{row.profile}: {row.score:.0%}")

with change_tab:
    display_changes = changes.copy()
    for column in ("baseline", "simulated", "change"):
        display_changes[column] = display_changes[column].map(lambda value: f"{value:+.0%}" if column == "change" else f"{value:.0%}")
    render_copyable_table(
        display_changes[["trait", "baseline", "simulated", "change"]].rename(
            columns={
                "trait": "Trait", "baseline": "Baseline", "simulated": "Simulated", "change": "Change"
            }
        ),
    )

with neighbour_tab:
    neighbour_display = neighbours.copy()
    neighbour_display["Record"] = neighbour_display.apply(
        lambda row: f"{int(row['wins'])}-{int(row['games']-row['wins'])}", axis=1
    )
    neighbour_display["Similarity"] = neighbour_display["similarity"].map(
        lambda value: f"{value:.1%}"
    )
    neighbour_display["NetRtg"] = neighbour_display["net_rating"].map(
        lambda value: f"{value:+.1f}"
    )
    render_copyable_table(
        neighbour_display[
            ["season", "team", "team_abbreviation", "Record", "NetRtg", "league_rank", "Similarity"]
        ].rename(
            columns={
                "season": "Season", "team": "Team", "team_abbreviation": "Abbr", "league_rank": "Rank"
            }
        ),
    )
    st.caption(
        "Neighbours are retrieved from standardized roster embeddings, not team names or outcomes. "
        + (
            "This roster is beyond the empirical support boundary, so even the closest rows are analogues, not validation for a forecast."
            if not quality_supported
            else ""
        )
    )

with ai_tab:
    st.write(
        "The numeric model, roster changes, profile scores and historical neighbours are already fixed. "
        "A deterministic template translates this evidence packet into cited analysis."
    )
    report_key = (
        f"roster_report_{season}_{team}_{start_mode}_"
        f"{hash((tuple(sorted(final_ids)), tuple(sorted(forced_ids))))}"
    )
    if st.button("Explain this roster", type="primary"):
        with st.spinner("Explaining the fixed simulation…"):
            st.session_state[report_key] = generate_roster_simulation_report(packet)
    if report_key in st.session_state:
        result = st.session_state[report_key]
        st.markdown(result.text)
        st.download_button(
            "Download simulation as Markdown",
            result.text,
            file_name=f"{team.lower()}_{season}_roster_simulation.md",
            mime="text/markdown",
        )

with evidence_tab:
    st.subheader("Validation scope")
    render_copyable_table(
        pd.DataFrame(
            [
                {
                    "Layer": "Observed team-profile model",
                    "Status": "Validated",
                    "Evidence": f"Leave-one-season-out MAE {evaluation['selected_model_mae']:.2f}",
                },
                {
                    "Layer": "Historical support guard",
                    "Status": "Validated behavior",
                    "Evidence": "95th percentile of leave-one-out neighbour distances",
                },
                {
                    "Layer": "Automatic 240-minute rotation",
                    "Status": "Invariant tested",
                    "Evidence": "Order-independent; position overlap, lower rotation impact, exact 240 total",
                },
                {
                    "Layer": "Net Rating → expected wins",
                    "Status": "Validated calibration",
                    "Evidence": (
                        f"Bounded logistic LOSO MAE "
                        f"{evaluation['win_calibration']['bounded_logistic_mae']:.2f} wins "
                        f"vs {evaluation['win_calibration']['linear_mae']:.2f} linear"
                    ),
                },
                {
                    "Layer": "Talent-upside direction",
                    "Status": "Constrained behavior",
                    "Evidence": "Non-negative rotation and top-three talent coefficients",
                },
                {
                    "Layer": "Real transaction outcomes",
                    "Status": "Not yet validated",
                    "Evidence": "Requires prior-season, time-aware addition/removal backtests",
                },
                {
                    "Layer": "Lineup fit and chemistry",
                    "Status": "Not modeled",
                    "Evidence": "Requires lineup possessions, roles and interaction features",
                },
            ]
        ),
    )
    st.caption(
        "A low team-profile MAE does not validate the causal effect of adding a player. "
        "BallDNA keeps those claims separate rather than borrowing confidence from the wrong test."
    )
    model_rows = pd.DataFrame(evaluation["models"])
    st.subheader("Model comparison")
    render_copyable_table(
        pd.DataFrame(
            {
                "Model": model_rows["model"],
                "NetRtg MAE": model_rows["mae"].map(lambda value: f"{value:.2f}"),
                "RMSE": model_rows["rmse"].map(lambda value: f"{value:.2f}"),
                "Rank correlation": model_rows["rank_correlation"].map(lambda value: f"{value:.3f}"),
            }
        ),
    )
    st.success(
        f"Selected {evaluation['selected_model']} on leave-one-season-out validation: "
        f"{evaluation['selected_model_mae']:.2f} Net Rating MAE."
    )
    st.subheader("Extrapolation guard")
    st.write(
        f"Nearest historical distance: **{support['nearest_distance']:.2f}** · "
        f"95th-percentile support boundary: **{support['support_threshold']:.2f}** · "
        f"status: **{'supported' if quality_supported else 'outside historical support'}**."
    )
    if support["features_outside_training_range"]:
        violation_rows = pd.DataFrame(support["features_outside_training_range"])
        render_copyable_table(
            violation_rows.rename(
                columns={
                    "feature": "Feature",
                    "value": "Roster value",
                    "training_min": "Historical min",
                    "training_max": "Historical max",
                    "direction": "Outside direction",
                }
            ),
        )
    st.subheader("Evidence packet used by the summary")
    st.json(packet, expanded=False)

render_build_trace(
    [
        ("Player features", "Engineer per-36 production, shrunk shooting accuracy, shot zones/actions, playmaking, rebounding, and defensive box-score proxies."),
        ("Player embedding", "Convert 13 player traits to within-season percentiles so unlike statistics share one interpretable vector space."),
        ("Role and reliability", "Keep style per-36, then combine rate production, observed-role context and a sample-shrunk plus/minus context. The latter helps distinguish trusted rotation impact from garbage-time efficiency, but is not treated as causal player value."),
        ("Roster pooling", "Use observed player-team stint MPG as the typical-game role, compress the smallest roles to an exact 240 minutes with a 48 MPG cap, then combine player vectors with a fixed 65% rotation / 35% top-three talent index."),
        ("Model selection", f"Compare four learned regressors and a baseline across held-out seasons; select {evaluation['selected_model']} by Net Rating MAE."),
        ("Support gate", "Compare the simulated roster's nearest-neighbour distance with the 95th-percentile historical boundary; withhold quality forecasts for unsupported extrapolations."),
        ("Counterfactual", "Start from the observed 240-minute rotation. Source-team MPG is only a ceiling: additions earn minutes from lower-impact incumbents at overlapping positions, strongest additions are allocated first, and removals open minutes for redistribution."),
        ("Expected wins", "Fit a separate non-negative talent model, anchor its change to the real team baseline, then use a bounded logistic Net Rating-to-wins calibration so elite-team gains saturate instead of growing linearly."),
        ("Retrieval", "Find the nearest historical team-seasons in standardized roster-embedding space."),
        ("Rule-based summary", "Translate fixed estimates, changes and neighbours into a cited deterministic explanation."),
    ]
)
st.warning(
    "Interpretation boundary: this is a descriptive counterfactual. It does not model contracts, "
    "injuries, transaction feasibility, coaching, chemistry, or causal lineup interactions."
)
st.caption(f"Active model input: {len(MODEL_FEATURES)} roster features over {evaluation['training_rows']} team-seasons.")
