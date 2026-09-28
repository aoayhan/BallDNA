"""Two-player comparison page."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from _bootstrap import initialize_app  # noqa: E402
from components.charts import comparison_bar_chart, similarity_bucket  # noqa: E402
from components.styles import apply_styles  # noqa: E402
from components.ui import (  # noqa: E402
    player_selector,
    render_build_trace,
    render_copyable_table,
)
from ball_ai.analytics.tables import (  # noqa: E402
    basketball_reference_season_table,
    shot_profile_comparison_table,
)
from ball_ai.analytics.play_style import (  # noqa: E402
    BROAD_OFFENSIVE_FEATURES,
    MODERN_MOVE_FEATURES,
    OFFENSIVE_PRESENCE_WEIGHT,
    SIAMESE_DETAIL_START,
    SIAMESE_ENSEMBLE_WEIGHT,
    TEMPORAL_ENSEMBLE_WEIGHT,
    find_style_neighbors,
    input_feature_comparison,
)
from ball_ai.analytics.siamese_attribution import (  # noqa: E402
    format_attribution_table,
    load_siamese_inference_artifacts,
    siamese_pair_attribution,
)
from ball_ai.config import settings  # noqa: E402
from ball_ai.data.database import (  # noqa: E402
    get_player_profile,
    get_player_shot_profile,
    get_players,
)
from ball_ai.data.historical_store import (  # noqa: E402
    get_latest_historical_player_teams,
    get_player_season_history,
    get_player_season_team,
    get_player_shot_history,
)


@st.cache_data(show_spinner=False)
def _load_style_tables(root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(root)
    embeddings = pd.read_parquet(path / "player_style_embeddings.parquet")
    features = pd.read_parquet(path / "player_style_features.parquet")
    impact_path = path / "darko_dpm.parquet"
    if impact_path.exists():
        impact = pd.read_parquet(
            impact_path, columns=["player_id", "season", "dpm", "o_dpm", "d_dpm"]
        ).drop_duplicates(["player_id", "season"], keep="last")
        for frame in (embeddings, features):
            frame.drop(columns=["dpm", "o_dpm", "d_dpm"], errors="ignore", inplace=True)
        embeddings = embeddings.merge(impact, on=["player_id", "season"], how="left")
        features = features.merge(impact, on=["player_id", "season"], how="left")
    return embeddings, features


@st.cache_resource(show_spinner=False)
def _load_style_models(root: str) -> dict:
    return joblib.load(Path(root) / "play_style_models.joblib")


@st.cache_resource(show_spinner=False)
def _load_temporal_model(root: str) -> dict | None:
    path = Path(root) / "model_registry/play_style/candidates/temporal_contrastive_v1/model.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def _load_siamese_embeddings(root: str) -> pd.DataFrame | None:
    path = Path(root) / "siamese_offensive_embeddings.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_resource(show_spinner=False)
def _load_siamese_attribution_artifacts(path: str) -> dict:
    return load_siamese_inference_artifacts(Path(path))


@st.cache_data(show_spinner=False)
def _load_impact(root: str) -> pd.DataFrame:
    return pd.read_parquet(
        Path(root) / "darko_dpm.parquet",
        columns=["player_id", "season", "dpm", "o_dpm"],
    ).drop_duplicates(["player_id", "season"], keep="last")


@st.cache_data(show_spinner=False)
def _comparison_players(root: str) -> pd.DataFrame:
    path = Path(root) / "player_style_embeddings.parquet"
    if not path.exists():
        return get_players()
    history = pd.read_parquet(path, columns=[
        "player_id", "player_name", "season", "position", "profile", "lens",
        "method", "eligible",
    ])
    history = history.loc[
        history["profile"].eq("Broad history")
        & history["lens"].eq("Offensive")
        & history["method"].eq("Denoising autoencoder")
        & history["eligible"].astype("boolean").fillna(False)
    ]
    players = history.sort_values("season").drop_duplicates("player_id", keep="last")[[
        "player_id", "player_name", "position",
    ]]
    latest_teams = get_latest_historical_player_teams(Path(root))
    players = players.merge(latest_teams, on="player_id", how="left", validate="one_to_one")
    current_teams = get_players().set_index("player_id")["team"]
    players["display_team"] = players["player_id"].map(current_teams)
    recent_seasons = sorted(history["season"].dropna().unique())[-2:]
    players["display_team"] = players["display_team"].fillna(
        players["team"].where(players["latest_season"].isin(recent_seasons))
    )
    return players.sort_values("player_name").reset_index(drop=True)


def _comparison_profile(
    player_id: int, players: pd.DataFrame, season: str | None = None
) -> dict | None:
    if season is None:
        profile = get_player_profile(player_id)
        if profile is not None:
            return profile
    history = get_player_season_history(player_id)
    if history.empty:
        return None
    selected = history if season is None else history.loc[history["season"].eq(season)]
    if selected.empty:
        return None
    profile = selected.iloc[-1].to_dict()
    games = float(profile.get("games_played") or 0)
    profile["two_points_made_per_game"] = (
        float(profile.get("two_points_made_total") or 0) / games if games else float("nan")
    )
    profile["two_point_attempts_per_game"] = (
        float(profile.get("two_point_attempts_total") or 0) / games if games else float("nan")
    )
    metadata = players.loc[players["player_id"].astype(int).eq(int(player_id))].iloc[0]
    team = get_player_season_team(player_id, str(profile["season"]))
    profile.update({"team": team or metadata.get("team", "N/A"), "position": metadata["position"]})
    return profile


def _comparison_shot_profile(player_id: int, season: str | None = None) -> dict | None:
    if season is None:
        profile = get_player_shot_profile(player_id)
        if profile is not None:
            return profile
    history = get_player_shot_history(player_id)
    selected = history if season is None else history.loc[history["season"].eq(season)]
    return None if selected.empty else selected.iloc[-1].to_dict()


def _signed(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "—" if pd.isna(number) else f"{number:+.2f}"


def _feature_table(frame: pd.DataFrame, first_name: str, second_name: str) -> pd.DataFrame:
    display = frame.drop(columns=["Weak shared-absence evidence"], errors="ignore").rename(
        columns={"Reference": first_name, "Match": second_name}
    ).copy()
    for column in display.columns[1:]:
        display[column] = pd.to_numeric(display[column], errors="coerce").map(
            lambda value: "—" if pd.isna(value) else f"{value:.3f}"
        )
    return display


def _render_style_matchup(
    first_id: int,
    second_id: int,
    first_name: str,
    second_name: str,
) -> tuple[str, str] | None:
    root = settings.historical_data_dir
    required = (
        root / "player_style_embeddings.parquet",
        root / "player_style_features.parquet",
        root / "play_style_models.joblib",
    )
    if not all(path.exists() for path in required):
        st.info("Build the Player DNA assets to compare play style.")
        return

    embeddings, features = _load_style_tables(str(root))
    models = _load_style_models(str(root))
    temporal_model = _load_temporal_model(str(root))
    siamese_embeddings = _load_siamese_embeddings(str(root))
    attribution_path = settings.root_dir / "models/play_style/siamese_tabular_v4_inference.npz"
    siamese_attribution_artifacts = (
        _load_siamese_attribution_artifacts(str(attribution_path))
        if attribution_path.exists()
        else None
    )
    control_columns = st.columns(2)
    profile = control_columns[0].selectbox(
        "Profile depth", ["Broad history", "Modern detailed"], key="compare_style_profile"
    )
    lens = control_columns[1].selectbox(
        "Comparison lens", ["Offensive", "Overall", "Defensive"], key="compare_style_lens"
    )
    if lens == "Defensive":
        st.error(
            "Defensive Player DNA is still under development and should not be trusted by itself. "
            "Verify it with film, tracking data, and impact context."
        )
    method = "Denoising autoencoder"
    cohort = embeddings.loc[
        embeddings["profile"].eq(profile)
        & embeddings["lens"].eq(lens)
        & embeddings["method"].eq(method)
        & embeddings["eligible"].astype("boolean").fillna(False)
    ]
    first_seasons = sorted(
        cohort.loc[cohort["player_id"].astype(int).eq(int(first_id)), "season"].unique(),
        reverse=True,
    )
    second_seasons = sorted(
        cohort.loc[cohort["player_id"].astype(int).eq(int(second_id)), "season"].unique(),
        reverse=True,
    )
    if not first_seasons or not second_seasons:
        st.info(f"Both players need an eligible {profile.lower()} {lens.lower()} season.")
        return

    season_columns = st.columns(2)
    first_season = season_columns[0].selectbox(
        f"{first_name} season", first_seasons, key=f"compare_style_a_{first_id}_{profile}_{lens}"
    )
    second_season = season_columns[1].selectbox(
        f"{second_name} season", second_seasons, key=f"compare_style_b_{second_id}_{profile}_{lens}"
    )
    temporal_active = profile == "Broad history" and lens == "Offensive" and temporal_model is not None
    siamese_active = (
        profile == "Broad history" and lens == "Offensive" and siamese_embeddings is not None
    )
    try:
        match = find_style_neighbors(
            embeddings,
            first_id,
            lens=lens,
            method=method,
            season=first_season,
            profile=profile,
            top_n=1,
            candidate_player_id=second_id,
            candidate_season_start=second_season,
            candidate_season_end=second_season,
            features=features,
            artifact=models[profile][lens],
            presence_weight=(
                OFFENSIVE_PRESENCE_WEIGHT if lens == "Offensive" and not siamese_active else 0.0
            ),
            temporal_artifact=temporal_model if temporal_active and not siamese_active else None,
            temporal_weight=(
                TEMPORAL_ENSEMBLE_WEIGHT if temporal_active and not siamese_active else 0.0
            ),
            siamese_embeddings=siamese_embeddings if siamese_active else None,
            siamese_weight=SIAMESE_ENSEMBLE_WEIGHT if siamese_active else 0.0,
        ).iloc[0]
    except ValueError as exc:
        st.info(str(exc))
        return

    score = float(match["similarity_score"])
    st.metric("Similarity index", f"{score:.2%} · {similarity_bucket(score)}")
    st.caption(
        "Higher means more similar basketball tendencies—not equal ability, quality, or a probability. "
        "Selected-season DPM and O-DPM are shown below and do not increase the Style Twin score."
    )

    comparison = input_feature_comparison(
        features,
        models[profile][lens],
        first_id,
        second_id,
        first_season,
        second_season,
        feature_names=(
            (
                BROAD_OFFENSIVE_FEATURES
                if min(first_season, second_season) < SIAMESE_DETAIL_START
                else [*BROAD_OFFENSIVE_FEATURES, *MODERN_MOVE_FEATURES]
            )
            if siamese_active
            else None
        ),
    )
    attribution = None
    if siamese_active and siamese_attribution_artifacts is not None:
        _, attribution = siamese_pair_attribution(
            features,
            siamese_attribution_artifacts,
            first_id,
            second_id,
            first_season,
            second_season,
        )
    shared_tab, differences_tab, context_tab = st.tabs(
        ["Shared tendencies", "Largest differences", "Impact and efficiency"]
    )
    with shared_tab:
        if attribution is not None:
            st.markdown("**Model-derived similarity drivers**")
            render_copyable_table(format_attribution_table(
                attribution, "Similarity contribution", first_name, second_name, limit=12
            ))
            st.caption(
                "Score drop when either player's feature is neutralized. Positive values indicate "
                "behaviors that pull this pair together in the Siamese embedding."
            )
            st.markdown("**Observed shared tendencies**")
        render_copyable_table(
            _feature_table(
                comparison.sort_values(
                    ["Shared tendency strength", "Standardized gap"], ascending=[False, True]
                ).head(12),
                first_name,
                second_name,
            )
        )
    with differences_tab:
        if attribution is not None:
            st.markdown("**Model-derived difference drivers**")
            render_copyable_table(format_attribution_table(
                attribution, "Difference penalty", first_name, second_name, limit=12
            ))
            st.caption(
                "Score gain when either player's value is matched to the other. Positive values "
                "indicate behaviors that push this pair apart; effects are symmetric and non-additive."
            )
            st.markdown("**Observed largest differences**")
        render_copyable_table(
            _feature_table(
                comparison.sort_values("Standardized gap", ascending=False).head(12),
                first_name,
                second_name,
            )
        )
    with context_tab:
        rows = features.loc[
            (features["player_id"].astype(int).eq(int(first_id)) & features["season"].eq(first_season))
            | (features["player_id"].astype(int).eq(int(second_id)) & features["season"].eq(second_season))
        ].copy()
        rows["Player"] = rows.apply(
            lambda row: first_name if int(row["player_id"]) == int(first_id) else second_name,
            axis=1,
        )
        display = rows[[
            "Player", "season", "age", "games_played", "shot_attempts", "dpm", "o_dpm", "d_dpm",
            "effective_field_goal_percentage", "field_goal_percentage", "three_point_percentage",
            "free_throw_attempt_rate",
        ]].rename(columns={
            "season": "Season", "age": "Age", "games_played": "G", "shot_attempts": "Shot attempts",
            "dpm": "DPM", "o_dpm": "O-DPM", "d_dpm": "D-DPM",
            "effective_field_goal_percentage": "eFG%", "field_goal_percentage": "FG%",
            "three_point_percentage": "3P%", "free_throw_attempt_rate": "FTr",
        })
        for column in ("DPM", "O-DPM", "D-DPM"):
            display[column] = display[column].map(_signed)
        for column in ("eFG%", "FG%", "3P%", "FTr"):
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda value: "—" if pd.isna(value) else f"{value:.1%}"
            )
        render_copyable_table(display)

    formula = (
        "coverage-aware Siamese Player DNA: 50% stable + 50% detailed when both seasons have 2007-08+ move coverage"
        if siamese_active
        else "42% denoising Player DNA + 30% temporal metric learning + 26.6% positive behavior + 1.4% shared absence"
        if temporal_active
        else "60% denoising Player DNA + 38% positive behavior + 2% shared absence"
        if lens == "Offensive"
        else "angular similarity between normalized denoising-autoencoder Player DNA vectors"
    )
    st.caption(f"How this was built: {formula}. Player identity, team, position, height, and weight are excluded.")
    return first_season, second_season


st.set_page_config(page_title="Compare · BallDNA", page_icon="⚖️", layout="wide")
apply_styles()
initialize_app()
st.title("Compare Players")
st.caption(
    "Compare two structured profiles, learned Player DNA, and separate impact context."
)

players = _comparison_players(str(settings.historical_data_dir))
left, right = st.columns(2)
with left:
    first_id = player_selector(
        players,
        "Player A",
        "compare_a",
        default_player_name="Luka Doncic",
        single_box=True,
    )
with right:
    second_id = player_selector(
        players,
        "Player B",
        "compare_b",
        default_player_name="James Harden",
        single_box=True,
    )

if first_id is None or second_id is None:
    st.stop()
if first_id == second_id:
    st.warning("Choose two different players to build a comparison.")
    st.stop()

player_names = players.set_index("player_id")["player_name"]
with st.expander("Player DNA style matchup", expanded=True):
    selected_seasons = _render_style_matchup(
        first_id,
        second_id,
        str(player_names.loc[first_id]),
        str(player_names.loc[second_id]),
    )
if selected_seasons is None:
    st.stop()
first_season, second_season = selected_seasons
first = _comparison_profile(first_id, players, first_season)
second = _comparison_profile(second_id, players, second_season)
if first is None or second is None:
    st.error("Season statistics are unavailable for one of the selected players.")
    st.stop()
first_shots = _comparison_shot_profile(first_id, first_season)
second_shots = _comparison_shot_profile(second_id, second_season)
impact = _load_impact(str(settings.historical_data_dir)).set_index(["player_id", "season"])
for profile in (first, second):
    key = (int(profile["player_id"]), str(profile["season"]))
    if key in impact.index:
        profile.update(impact.loc[key, ["dpm", "o_dpm"]].to_dict())
    else:
        profile.update({"dpm": float("nan"), "o_dpm": float("nan")})

st.plotly_chart(
    comparison_bar_chart(first, second),
    width="stretch",
    config={"scrollZoom": False},
)
st.caption(
    "Drag or swipe horizontally to inspect more per-game statistics in Basketball Reference order."
)
render_copyable_table(basketball_reference_season_table([first, second]))
if first_shots and second_shots:
    with st.expander("Compare engineered shot fingerprints", expanded=True):
        render_copyable_table(
            shot_profile_comparison_table([
                (first["player_name"], first_shots),
                (second["player_name"], second_shots),
            ])
        )
        st.caption(
            "Players are rows; shot-selection features run left-to-right. Difficulty is "
            "a zone/action expected-FG proxy, not direct defender tracking."
        )

render_build_trace(
    [
        ("Parquet retrieval", "Load the two selected-season profiles."),
        ("Horizontal comparison", "Present players as rows and stats left-to-right in Basketball Reference order."),
        ("Player DNA", "Score behavioral similarity with the validated learned representation."),
        ("Impact separation", "Show DARKO impact beside style without allowing quality to inflate similarity."),
    ]
)
