"""Explainable, role-aware player retrieval page."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from _bootstrap import initialize_app  # noqa: E402
from components.charts import (  # noqa: E402
    SIMILARITY_BUCKETS,
    similarity_chart,
    similarity_radar_chart,
)
from components.styles import apply_styles  # noqa: E402
from components.ui import (  # noqa: E402
    player_selector,
    render_build_trace,
    render_copyable_table,
)
from ball_ai.ai.grounding import build_similarity_evidence  # noqa: E402
from ball_ai.ai.report_generator import generate_similarity_explanation  # noqa: E402
from ball_ai.analytics.similarity import (  # noqa: E402
    FEATURE_FAMILIES,
    ROLE_PRESETS,
    active_similarity_features,
    contribution_table,
    find_similar_players,
)
from ball_ai.analytics.play_style import (  # noqa: E402
    BROAD_OFFENSIVE_FEATURES,
    OFFENSIVE_PRESENCE_WEIGHT,
    SIAMESE_DETAIL_START,
    SIAMESE_ENSEMBLE_WEIGHT,
    TEMPORAL_ENSEMBLE_WEIGHT,
    find_style_neighbors,
    historical_self_similarities,
    input_feature_comparison,
    shot_sample_reliability,
    temporal_self_match,
)
from ball_ai.analytics.stat_order import BASKETBALL_REFERENCE_STAT_ORDER  # noqa: E402
from ball_ai.config import settings  # noqa: E402
from ball_ai.data.database import (  # noqa: E402
    get_data_status_text,
    get_player_profile,
    get_players,
    get_season_stats,
)
from ball_ai.data.historical_store import get_latest_historical_player_teams  # noqa: E402


st.set_page_config(page_title="Similar Players · BallDNA", page_icon="🧭", layout="wide")
apply_styles()
initialize_app()
st.markdown(
    """
    <style>
        div[data-testid="stMainBlockContainer"] {padding-top: .65rem !important;}
        h1 {margin-bottom: .2rem !important;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Similar Players")
st.caption(
    "A self-supervised encoder learns offensive, defensive, and overall Player DNA. "
    "Impact remains separate, so playing alike never means being equally good."
)


@st.cache_data(show_spinner=False)
def _load_style_tables(root: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = Path(root)
    embeddings = pd.read_parquet(path / "player_style_embeddings.parquet")
    features = pd.read_parquet(path / "player_style_features.parquet")
    impact_path = path / "darko_dpm.parquet"
    if impact_path.exists():
        impact = pd.read_parquet(
            impact_path, columns=["player_id", "season", "dpm", "o_dpm", "d_dpm"]
        ).drop_duplicates(["player_id", "season"], keep="last")
        embeddings = embeddings.drop(
            columns=["dpm", "o_dpm", "d_dpm"], errors="ignore"
        ).merge(impact, on=["player_id", "season"], how="left", validate="many_to_one")
        features = features.drop(
            columns=["dpm", "o_dpm", "d_dpm"], errors="ignore"
        ).merge(impact, on=["player_id", "season"], how="left", validate="one_to_one")
    evaluation = json.loads((path / "play_style_evaluation.json").read_text(encoding="utf-8"))
    return embeddings, features, evaluation


@st.cache_resource(show_spinner=False)
def _load_style_models(root: str) -> dict:
    return joblib.load(Path(root) / "play_style_models.joblib")


@st.cache_resource(show_spinner=False)
def _load_temporal_component(root: str) -> dict | None:
    path = Path(root) / "model_registry/play_style/candidates/temporal_contrastive_v1/model.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def _load_temporal_evaluation(root: str) -> dict | None:
    path = Path(root) / "model_registry/play_style/candidates/temporal_contrastive_v1/evaluation.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner=False)
def _load_siamese_embeddings(root: str) -> pd.DataFrame | None:
    path = Path(root) / "siamese_offensive_embeddings.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def _load_siamese_evaluation(root: str) -> dict | None:
    path = settings.root_dir / "models/play_style/siamese_tabular_v3_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner=False)
def _load_latest_player_teams(root: str) -> pd.DataFrame:
    return get_latest_historical_player_teams(Path(root))


def _advanced_retrieval_settings(
    style_available: bool,
    validated_method: str | None = None,
) -> None:
    with st.expander("Advanced retrieval settings"):
        st.radio(
            "Retrieval system",
            ["Trained Player DNA", "Transparent statistical baseline"],
            horizontal=True,
            disabled=not style_available,
            key="similar_retrieval_system",
            help="Player DNA is learned without player-comparison labels. The baseline remains available for auditing.",
        )
        st.radio(
            "Profile depth",
            ["Broad history", "Modern detailed"],
            horizontal=True,
            disabled=st.session_state["similar_retrieval_system"] != "Trained Player DNA",
            key="similar_profile_depth",
            help="Broad history uses consistent cross-era inputs. Modern detailed adds 2020+ possession actions.",
        )
        st.selectbox(
            "Embedding model",
            ["Denoising autoencoder", "Cosine baseline", "PCA baseline"],
            disabled=st.session_state["similar_retrieval_system"] != "Trained Player DNA",
            key="similar_embedding_model",
            help=f"Validated winner for the active profile and lens: {validated_method or 'unavailable'}.",
        )

style_root = settings.historical_data_dir
style_available = all(
    (style_root / filename).exists()
    for filename in (
        "player_style_embeddings.parquet",
        "player_style_features.parquet",
        "play_style_evaluation.json",
        "play_style_models.joblib",
    )
)
st.session_state.setdefault("similar_retrieval_system", "Trained Player DNA")
st.session_state.setdefault("similar_profile_depth", "Broad history")
st.session_state.setdefault("similar_embedding_model", "Denoising autoencoder")
model_mode = st.session_state["similar_retrieval_system"]
embeddings = style_features = model_evaluation = artifacts = None
if not style_available:
    st.warning("Player DNA assets are not built yet; using the transparent baseline.")
    model_mode = "Transparent statistical baseline"
    st.session_state["similar_retrieval_system"] = model_mode

if model_mode == "Trained Player DNA":
    embeddings, style_features, model_evaluation = _load_style_tables(str(style_root))
    artifacts = _load_style_models(str(style_root))
    temporal_artifact = _load_temporal_component(str(style_root))
    temporal_evaluation = _load_temporal_evaluation(str(style_root))
    siamese_embeddings = _load_siamese_embeddings(str(style_root))
    siamese_evaluation = _load_siamese_evaluation(str(style_root))
    profile = st.session_state["similar_profile_depth"]
    method_options = ["Denoising autoencoder", "Cosine baseline", "PCA baseline"]
    lens_options = ["Offensive", "Overall", "Defensive"]
    method = st.session_state.get("similar_embedding_model", method_options[0])
    if method not in method_options:
        method = method_options[0]
        st.session_state["similar_embedding_model"] = method

    reference_history = embeddings.loc[
        embeddings["profile"].eq(profile)
        & embeddings["method"].eq(method)
        & embeddings["eligible"].astype("boolean").fillna(False)
    ]
    latest_teams = _load_latest_player_teams(str(style_root))
    players = (
        reference_history.sort_values("season")
        .drop_duplicates("player_id", keep="last")
        [["player_id", "player_name", "position"]]
        .merge(
            latest_teams,
            on="player_id",
            how="left",
            validate="one_to_one",
        )
    )
    players["team"] = players["team"].fillna("N/A")
    current_teams = get_players().set_index("player_id")["team"]
    players["display_team"] = players["player_id"].map(current_teams)
    recent_seasons = sorted(reference_history["season"].dropna().unique())[-2:]
    recently_active = players["latest_season"].isin(recent_seasons)
    players["display_team"] = players["display_team"].fillna(
        players["team"].where(recently_active)
    )
    players = players[
        ["player_id", "player_name", "team", "position", "display_team"]
    ].sort_values("player_name").reset_index(drop=True)
    if players.empty:
        st.error(f"No players have an eligible {profile.lower()} season.")
        st.stop()
else:
    players = get_players()

player_id = player_selector(
    players,
    "Reference player",
    "similar_player",
    default_player_name="Shai Gilgeous-Alexander",
    compact=True,
    show_count=False,
)
if player_id is None:
    st.stop()
if model_mode == "Trained Player DNA":
    player_lenses = set(
        reference_history.loc[
            reference_history["player_id"].astype(int).eq(int(player_id)), "lens"
        ].dropna()
    )
    available_lenses = [lens_name for lens_name in lens_options if lens_name in player_lenses]
    if not available_lenses:
        st.error("No eligible Player DNA season is available for this player.")
        st.stop()
    lens = st.session_state.get("similar_comparison_lens", available_lenses[0])
    if lens not in available_lenses:
        lens = available_lenses[0]
        st.session_state["similar_comparison_lens"] = lens
    lens_column, _ = st.columns([.8, 2.2])
    with lens_column:
        lens = st.selectbox(
            "Comparison lens",
            available_lenses,
            key="similar_comparison_lens",
            help="Each lens has its own trained encoder and eligibility rule.",
        )
    if len(available_lenses) < len(lens_options):
        st.caption(
            "Only eligible lenses are shown. Overall and Defensive require NBA matchup "
            "tracking available from 2017–18."
        )
    validated_method = model_evaluation.get("selected_methods", {}).get(profile, {}).get(lens)
    eligible_history = reference_history.loc[reference_history["lens"].eq(lens)]

season = get_season_stats()
active_features = active_similarity_features(season)
shot_features_active = bool(season["shot_attempts"].notna().any())

if model_mode == "Trained Player DNA":
    primary_controls = st.columns([1.2, .6, .6])
    available_seasons = sorted(eligible_history["season"].dropna().unique())
    reference_seasons = sorted(
        eligible_history.loc[
            eligible_history["player_id"].astype(int).eq(int(player_id)), "season"
        ].dropna().unique()
    )
    if not reference_seasons:
        st.error(f"No eligible {lens.lower()} season is available for this player.")
        st.stop()
    default_reference_season = reference_seasons[-1]
    with primary_controls[1]:
        reference_season = st.selectbox(
            "Player season",
            reference_seasons,
            index=reference_seasons.index(default_reference_season),
            key=f"similar_reference_season_{player_id}_{profile}_{lens}_{method}",
            help="The player's latest eligible completed season is selected by default.",
        )
    with primary_controls[2]:
        unique_players = st.toggle(
            "Unique players",
            value=False,
            key="similar_unique_players",
            help="Keep only the highest-ranked season for each matching player.",
        )
    impact_column = {"Offensive": "o_dpm", "Defensive": "d_dpm", "Overall": "dpm"}[lens]
    impact_label = {"Offensive": "O-DPM", "Defensive": "D-DPM", "Overall": "DPM"}[lens]
    reference_row = eligible_history.loc[
        eligible_history["player_id"].astype(int).eq(int(player_id))
        & eligible_history["season"].eq(reference_season)
    ].iloc[0]
    reference_has_impact = pd.notna(pd.to_numeric(
        pd.Series([reference_row[impact_column]]), errors="coerce"
    ).iloc[0])
    comparison_options = ["Style Twin", "Comparable Player"] if reference_has_impact else ["Style Twin"]
    if st.session_state.get("similar_comparison_goal") not in comparison_options:
        st.session_state["similar_comparison_goal"] = "Style Twin"
    with primary_controls[0]:
        comparison_goal = st.radio(
            "Comparison goal",
            comparison_options,
            horizontal=True,
            key="similar_comparison_goal",
            help=(
                "Style Twin compares behavior only. Comparable Player limits candidates "
                "to a similar lens-specific DARKO impact band."
            ),
        )
    if not reference_has_impact:
        st.info(
            f"Comparable Player is unavailable for {reference_season}: no DARKO impact row "
            "is stored for this player-season. Style Twin remains available."
        )
    match_count_key = f"similar_match_count_{player_id}_{reference_season}_{profile}_{lens}_{method}"
    st.session_state.setdefault(match_count_key, 5)
    top_n = min(int(st.session_state[match_count_key]), 10)
    candidate_history = eligible_history
    if comparison_goal == "Comparable Player":
        candidate_history = candidate_history.loc[
            pd.to_numeric(candidate_history[impact_column], errors="coerce").notna()
        ]
    candidate_available_seasons = sorted(candidate_history["season"].dropna().unique())
    available_ages = pd.to_numeric(candidate_history["age"], errors="coerce").dropna().astype(int)
    position_aware = False
    impact_tolerance = None
    with st.expander("Refine candidate pool"):
        filters = st.columns([.75, 1.5, 1.5])
        with filters[0]:
            position_aware = st.toggle(
                "Position-aware",
                value=False,
                help="Optional broad G/F/C candidate filter. Position is never an encoder input.",
            )
        with filters[1]:
            candidate_seasons = st.select_slider(
                "Candidate seasons",
                options=candidate_available_seasons,
                value=(candidate_available_seasons[0], candidate_available_seasons[-1]),
                help="Controls which historical player-seasons may be returned.",
            )
        with filters[2]:
            candidate_ages = st.slider(
                "Candidate ages",
                min_value=int(available_ages.min()),
                max_value=int(available_ages.max()),
                value=(int(available_ages.min()), int(available_ages.max())),
            )
        if comparison_goal == "Comparable Player":
            impact_slider, impact_explanation = st.columns([1, 1.4])
            with impact_slider:
                impact_tolerance = st.slider(
                    "Maximum impact difference",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.5,
                    step=0.5,
                    help=f"Require candidates within this absolute {impact_label} difference.",
                )
            with impact_explanation:
                st.markdown("**Impact filtered by DARKO DPM**")
                st.caption(
                    f"Candidates must be within ±{impact_tolerance:.1f} {impact_label}. "
                    "Impact never enters Player DNA."
                )

    validation_note = "validated winner" if method == validated_method else f"validated winner: {validated_method}"
    if profile == "Broad history" and lens != "Offensive":
        st.caption(
            "Overall and defensive history begins in 2017–18 because matchup tracking is required; "
            "broad offense reaches 1996–97."
        )
    action_weight = model_evaluation.get("selected_action_weights", {}).get(profile, {}).get(lens)
    if profile == "Modern detailed" and lens != "Defensive":
        st.caption(
            f"Validated possession-action weight: {action_weight:.2f}."
            if action_weight
            else "Possession actions received zero validated weight for this lens."
        )
    temporal_active = (
        profile == "Broad history"
        and lens == "Offensive"
        and method == "Denoising autoencoder"
        and temporal_artifact is not None
    )
    siamese_active = (
        profile == "Broad history"
        and lens == "Offensive"
        and method == "Denoising autoencoder"
        and siamese_embeddings is not None
    )
    if siamese_active:
        retrieval_note = "coverage-aware Siamese Player DNA: stable cross-era + detailed 2007-08+"
    elif temporal_active:
        retrieval_note = (
            "validated temporal ensemble: 42% denoising Player DNA + 30% temporal metric "
            "learning + 26.6% positive behavior + 1.4% shared absence"
        )
    elif lens == "Offensive":
        retrieval_note = (
            "hybrid retrieval: 60% learned Player DNA + 38% positive behavior + "
            "2% information-weighted shared absence"
        )
    else:
        retrieval_note = "encoder retrieval"
    st.caption(
        f"{profile} · {lens} · {available_seasons[0]}–{available_seasons[-1]} · "
        f"{method} ({validation_note}) · {retrieval_note}."
    )
    method_test = next(
        row for row in model_evaluation["evaluations"]
        if row.get("profile") == profile and row.get("lens") == lens
        and row.get("method") == method and row.get("split") == "test"
    )
    if lens == "Defensive":
        st.error(
            "Defensive Player DNA is still under development and should not be trusted by itself. "
            f"Only {method_test['recall_at_5']:.1%} of held-out players retrieve their previous "
            "season in the top five without physical shortcuts. Verify it with film, tracking data, "
            "and impact context."
        )
    try:
        neighbors = find_style_neighbors(
            embeddings,
            player_id,
            lens=lens,
            method=method,
            season=reference_season,
            profile=profile,
            top_n=top_n,
            position_aware=position_aware,
            impact_tolerance=impact_tolerance,
            candidate_season_start=candidate_seasons[0],
            candidate_season_end=candidate_seasons[1],
            minimum_age=candidate_ages[0],
            maximum_age=candidate_ages[1],
            features=style_features,
            artifact=artifacts[profile][lens],
            presence_weight=(
                OFFENSIVE_PRESENCE_WEIGHT if lens == "Offensive" and not siamese_active else 0.0
            ),
            temporal_artifact=temporal_artifact if temporal_active and not siamese_active else None,
            temporal_weight=(
                TEMPORAL_ENSEMBLE_WEIGHT if temporal_active and not siamese_active else 0.0
            ),
            siamese_embeddings=siamese_embeddings if siamese_active else None,
            siamese_weight=SIAMESE_ENSEMBLE_WEIGHT if siamese_active else 0.0,
            unique_players=unique_players,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    details = neighbors.merge(
        style_features,
        on=["player_id", "season"],
        how="left",
        suffixes=("", "_stats"),
        validate="one_to_one",
    )
    reference_embedding = embeddings.loc[
        embeddings["player_id"].astype(int).eq(int(player_id))
        & embeddings["season"].eq(reference_season)
        & embeddings["profile"].eq(profile)
        & embeddings["lens"].eq(lens)
        & embeddings["method"].eq(method)
    ].iloc[0]
    reference_name = str(reference_embedding["player_name"])
    reference_stats = style_features.loc[
        style_features["player_id"].astype(int).eq(int(player_id))
        & style_features["season"].eq(reference_season)
    ].iloc[0]
    reference_impact = pd.to_numeric(
        pd.Series([reference_embedding[impact_column]]), errors="coerce"
    ).iloc[0]
    if siamese_active:
        try:
            self_history = find_style_neighbors(
                embeddings,
                player_id,
                lens=lens,
                method=method,
                season=reference_season,
                profile=profile,
                top_n=max(len(reference_seasons), 1),
                candidate_player_id=player_id,
                features=style_features,
                artifact=artifacts[profile][lens],
                presence_weight=0.0,
                temporal_artifact=None,
                temporal_weight=0.0,
                siamese_embeddings=siamese_embeddings,
                siamese_weight=SIAMESE_ENSEMBLE_WEIGHT,
            )
            self_history = (
                self_history.loc[self_history["season"].lt(reference_season)]
                .sort_values("season", ascending=False)
                .head(3)
            )
        except ValueError:
            self_history = pd.DataFrame()
    else:
        self_history = historical_self_similarities(
            embeddings,
            player_id,
            lens=lens,
            method=method,
            season=reference_season,
            profile=profile,
        )
    matches_tab, why_tab, context_tab, evaluation_tab = st.tabs(
        ["Matches", "Why they are similar", "Context splits", "Model evaluation"]
    )
    with matches_tab:
        st.plotly_chart(
            similarity_chart(
                details,
                impact_label=impact_label if comparison_goal == "Comparable Player" else None,
            ),
            width="stretch",
        )
        bucket_guide = []
        for lower, upper, label, color in SIMILARITY_BUCKETS:
            range_label = (
                f"<{upper:.0%}" if lower == 0
                else f"{lower:.0%}+" if upper > 1
                else f"{lower:.0%}–{upper - .01:.0%}"
            )
            bucket_guide.append(f":{color}-badge[{range_label} · {label}]")
        st.markdown(" ".join(bucket_guide))
        if comparison_goal == "Comparable Player":
            st.caption(
                f"Reference {impact_label}: {reference_impact:+.2f}. Signed labels show each "
                "candidate's impact difference: + is higher, − is lower."
            )
        if top_n < 10:
            _, more_column, _ = st.columns([1, .7, 1])
            if more_column.button(
                "Show 5 more matches",
                key=f"show_more_{match_count_key}",
                width="stretch",
            ):
                st.session_state[match_count_key] = 10
                st.rerun()
        stat_keys = [item.key for item in BASKETBALL_REFERENCE_STAT_ORDER]
        display = details[[
            "player_name", "season", "age", "position", "shot_attempts", *stat_keys,
            "similarity_score", "dpm", "o_dpm", "d_dpm", "impact_difference",
        ]].copy()
        display.insert(5, "reliability", display["shot_attempts"].map(shot_sample_reliability).map({
            "High": "🟢 High", "Medium": "🟠 Medium",
            "Limited": "⚪ Limited", "Unavailable": "—",
        }))
        for item in BASKETBALL_REFERENCE_STAT_ORDER:
            values = pd.to_numeric(display[item.key], errors="coerce")
            if item.value_format == "percentage":
                display[item.key] = values.map(lambda value: "—" if pd.isna(value) else f"{value:.1%}")
            elif item.value_format == "number":
                display[item.key] = values.map(lambda value: "—" if pd.isna(value) else f"{value:.1f}")
            else:
                display[item.key] = values.map(lambda value: "—" if pd.isna(value) else f"{int(value)}")
        display["similarity_score"] = display["similarity_score"].map(lambda value: f"{value:.1%}")
        for column in ("dpm", "o_dpm", "d_dpm", "impact_difference"):
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda value: "—" if pd.isna(value) else f"{value:+.2f}"
            )
        render_copyable_table(
            display.rename(columns={
                "player_name": "Player",
                "season": "Season",
                "age": "Age",
                "position": "Pos",
                "shot_attempts": "Shot attempts",
                "reliability": "Reliability",
                **{item.key: item.abbreviation for item in BASKETBALL_REFERENCE_STAT_ORDER},
                "similarity_score": "Player DNA index",
                "dpm": "DPM",
                "o_dpm": "O-DPM",
                "d_dpm": "D-DPM",
                "impact_difference": f"{impact_label} vs {reference_name}",
            }),
        )
        st.markdown(
            ":green-badge[High · 800+ attempts] "
            ":orange-badge[Medium · 400–799] "
            ":gray-badge[Limited · 100–399]"
        )
        st.caption(
            "Reliability describes how stable the neighbor list was under game resampling; "
            "it does not grade player quality."
        )
        if not self_history.empty:
            prior_seasons = "; ".join(
                f"{row.season}: {row.similarity_score:.1%}"
                for row in self_history.itertuples()
            )
            st.caption(
                f"Previous {reference_name} seasons are excluded from player results — "
                f"{prior_seasons}."
            )
        with st.expander("Impact and efficiency differences · not used for Style Twin ranking"):
            context = details[[
                "player_name", "season", "age", "similarity_score", impact_column,
                "effective_field_goal_percentage", "field_goal_percentage",
                "three_point_percentage",
            ]].copy()
            context["impact_gap"] = pd.to_numeric(
                context[impact_column], errors="coerce"
            ) - reference_impact
            for column in (
                "effective_field_goal_percentage", "field_goal_percentage", "three_point_percentage"
            ):
                reference_efficiency = pd.to_numeric(
                    pd.Series([reference_stats.get(column)]), errors="coerce"
                ).iloc[0]
                context[f"{column}_gap"] = (
                    pd.to_numeric(context[column], errors="coerce")
                    - reference_efficiency
                )
                context[column] = pd.to_numeric(context[column], errors="coerce").map(
                    lambda value: "—" if pd.isna(value) else f"{value:.1%}"
                )
                context[f"{column}_gap"] = context[f"{column}_gap"].map(
                    lambda value: "—" if pd.isna(value) else f"{value:+.1%}"
                )
            context["similarity_score"] = context["similarity_score"].map(lambda value: f"{value:.1%}")
            context[impact_column] = pd.to_numeric(context[impact_column], errors="coerce").map(
                lambda value: "—" if pd.isna(value) else f"{value:+.2f}"
            )
            context["impact_gap"] = context["impact_gap"].map(
                lambda value: "—" if pd.isna(value) else f"{value:+.2f}"
            )
            render_copyable_table(
                context.rename(columns={
                    "player_name": "Player",
                    "season": "Season",
                    "age": "Age",
                    "similarity_score": "Style index",
                    impact_column: impact_label,
                    "impact_gap": f"{impact_label} gap",
                    "effective_field_goal_percentage": "eFG%",
                    "effective_field_goal_percentage_gap": "eFG% gap",
                    "field_goal_percentage": "FG%",
                    "field_goal_percentage_gap": "FG% gap",
                    "three_point_percentage": "3P%",
                    "three_point_percentage_gap": "3P% gap",
                }),
            )
        _advanced_retrieval_settings(style_available, validated_method)

    with why_tab:
        labels = {
            (int(row.player_id), str(row.season)): f"{row.player_name} · {row.season} · {row.position}"
            for row in neighbors.itertuples()
        }
        compared_id, compared_season = st.selectbox(
            "Explain match", list(labels), format_func=labels.get
        )
        comparison = input_feature_comparison(
            style_features,
            artifacts[profile][lens],
            player_id,
            compared_id,
            reference_season,
            compared_season,
            feature_names=(
                BROAD_OFFENSIVE_FEATURES
                if siamese_active
                and min(reference_season, compared_season) < SIAMESE_DETAIL_START
                else None
            ),
        )
        close, different = st.columns(2)
        with close:
            st.markdown("**Strongest shared tendencies**")
            render_copyable_table(
                comparison.sort_values(
                    ["Shared tendency strength", "Standardized gap"],
                    ascending=[False, True],
                ).head(8),
            )
        with different:
            st.markdown("**Largest observed differences**")
            render_copyable_table(
                comparison.sort_values("Standardized gap", ascending=False).head(8)
            )
        st.caption(
            "Positive behavior provides strong evidence. Shared absence provides weak evidence, "
            "reduced further when that absence is common in training. Differences remain "
            "standardized input gaps, not invented meanings for latent dimensions."
        )

    with context_tab:
        row = style_features.loc[
            style_features["player_id"].astype(int).eq(int(player_id))
            & style_features["season"].eq(reference_season)
        ].iloc[0]
        context = pd.DataFrame([
            ("Home shot share", row.get("home_shot_frequency"), row.get("home_away_context_eligible")),
            ("Home/away zone-mix shift", row.get("home_away_zone_shift"), row.get("home_away_context_eligible")),
            ("First-half vs fourth-quarter shift", row.get("first_half_fourth_quarter_zone_shift"), row.get("quarter_context_eligible")),
            ("Regular season vs playoffs shift", row.get("regular_playoff_zone_shift"), row.get("playoff_context_eligible")),
            ("Early-clock shot share", row.get("early_clock_shot_rate"), row.get("timing_shots", 0) >= 75),
            ("Late-clock shot share", row.get("late_clock_shot_rate"), row.get("timing_shots", 0) >= 75),
        ], columns=["Context", "Value", "Eligible sample"])
        context["Value"] = pd.to_numeric(context["Value"], errors="coerce").map(
            lambda value: "—" if pd.isna(value) else f"{value:.3f}"
        )
        render_copyable_table(context)
        st.caption(
            "Home/away requires 40 shots in each split; quarter context requires 40 first-half and 40 fourth-quarter shots; playoff context requires 100 regular-season and 30 playoff shots. Early/late-clock behavior is a diagnostic reconstructed from possession changes and offensive-rebound resets; validation did not justify putting it in Player DNA yet."
        )

    with evaluation_tab:
        previous_seasons = [value for value in reference_seasons if value < reference_season]
        if previous_seasons:
            previous_season = previous_seasons[-1]
            try:
                self_match = temporal_self_match(
                    embeddings,
                    player_id,
                    lens=lens,
                    method=method,
                    query_season=reference_season,
                    candidate_season=previous_season,
                    profile=profile,
                )
                self_columns = st.columns(3)
                self_columns[0].metric(
                    "Encoder-only self rank",
                    f"#{self_match['rank']} of {self_match['candidate_count']}",
                )
                self_columns[1].metric("Self similarity", f"{self_match['similarity_score']:.1%}")
                self_columns[2].metric("Seasons", f"{reference_season} → {previous_season}")
            except ValueError as exc:
                st.info(f"Previous-season self-check unavailable: {exc}")
        else:
            st.info("No earlier eligible season is available for this player.")
        evaluation = pd.DataFrame(model_evaluation["evaluations"])
        evaluation = evaluation.loc[
            evaluation["profile"].eq(profile) & evaluation["lens"].eq(lens), [
            "method", "split", "eligible_queries", "recall_at_1", "recall_at_3", "recall_at_5",
            "recall_at_10", "mean_reciprocal_rank", "median_rank",
        ]]
        render_copyable_table(evaluation)
        if temporal_active and not siamese_active and temporal_evaluation:
            promoted_evidence = pd.concat([
                pd.DataFrame(temporal_evaluation["combined_holdout_summary"]).assign(
                    evaluation_set="19 holdout folds"
                ),
                pd.DataFrame(temporal_evaluation["established_rolling_summary"]).assign(
                    evaluation_set="8 established rolling folds"
                ),
            ], ignore_index=True)
            st.markdown("**Promoted temporal component · retrieval evidence**")
            render_copyable_table(promoted_evidence)
        if siamese_active and siamese_evaluation:
            siamese_evidence = pd.DataFrame([
                {"evaluation_set": "19 chronological holdouts", **values}
                for model, values in siamese_evaluation["all_19_holdouts"].items()
            ]).assign(model=["Stable Siamese", "Coverage-aware ensemble"])
            st.markdown("**Promoted Siamese component · retrieval evidence**")
            render_copyable_table(siamese_evidence)
        st.markdown(
            "These rows evaluate the encoder before the offensive presence-aware reranker. "
            "The model uses eligible history through 2022–23; later seasons are reserved for validation and test."
        )
        render_build_trace([
            ("Eligibility", model_evaluation["eligibility"][profile][lens.lower()]),
            ("Profile depth", "Broad History uses cross-era common inputs; Modern Detailed requires the richer 2020–present action feed."),
            ("Shortcut exclusions", "Player identity, team, position, height, weight, efficiency, and impact never enter the encoder."),
            ("Self-supervised task", "Corrupt each eligible player-season vector and reconstruct the clean behavioral inputs through a learned bottleneck."),
            ("Action data", "Excluded from Broad History; Modern Detailed uses assisted makes, fast breaks, second chances, and turnover-created shots."),
            ("Vector retrieval", "L2-normalize Player DNA and calculate cosine similarity."),
            ("Temporal metric learning", "Learn persistent behavior from repeated seasons: same-player seasons define positive classes and other player-seasons define negatives; identity never enters the input vector."),
            ("Siamese metric learning", "Send both player-seasons through the same MLP and use contrastive loss to pull adjacent same-player seasons together while pushing other players apart."),
            ("Coverage guard", "Exclude floater, pull-up, and step-back labels from Broad History because the source does not encode them before 2007-08."),
            ("Deployed offensive ensemble", "Use stable Siamese Player DNA for every pair; when both seasons are 2007-08 or later, blend 25% stable and 75% detailed Siamese similarity."),
            ("Impact separation", "Attach DPM, O-DPM, and D-DPM after retrieval; they never enter the style encoder."),
            ("Model selection", "Choose latent size, temperature, and the covered-pair blend on development folds, then report performance on separate chronological holdouts."),
        ], expanded=True)
    st.stop()

controls = st.columns([1.4, .9, .85, .9])
with controls[0]:
    preset = st.selectbox(
        "Role lens",
        list(ROLE_PRESETS),
        help="Changes feature importance for retrieval; it does not score player quality.",
    )
with controls[1]:
    position_aware = st.toggle(
        "Position-aware",
        value=True,
        help="Keep candidates that share at least one broad G/F/C label.",
    )
with controls[2]:
    minimum_games = st.number_input(
        "Minimum games",
        min_value=0,
        max_value=82,
        value=15,
        step=5,
    )
with controls[3]:
    minimum_shots = st.number_input(
        "Minimum shots",
        min_value=0,
        max_value=2000,
        value=100 if shot_features_active else 0,
        step=25,
        disabled=not shot_features_active,
        help="Reliability filter for the shot-style embedding.",
    )
st.caption(ROLE_PRESETS[preset]["description"])
baseline_match_count_key = f"baseline_match_count_{player_id}_{preset}"
st.session_state.setdefault(baseline_match_count_key, 5)
top_n = min(int(st.session_state[baseline_match_count_key]), 10)

reference = get_player_profile(player_id)
if reference is None:
    st.error("No season profile is available for the reference player.")
    st.stop()

try:
    neighbors = find_similar_players(
        season,
        player_id,
        top_n,
        preset=preset,
        position_aware=position_aware,
        minimum_games=int(minimum_games),
        minimum_shots=int(minimum_shots),
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

packet = build_similarity_evidence(
    reference,
    neighbors,
    preset,
    position_aware,
    int(minimum_games),
    get_data_status_text(),
    minimum_shots=int(minimum_shots),
)

matches_tab, why_tab, ai_tab, trace_tab = st.tabs(
    ["Matches", "Why players are similar", "Rule-based summary", "How this was built"]
)

with matches_tab:
    st.plotly_chart(similarity_chart(neighbors), width="stretch")
    if top_n < 10:
        _, more_column, _ = st.columns([1, .7, 1])
        if more_column.button(
            "Show 5 more matches",
            key=f"show_more_{baseline_match_count_key}",
            width="stretch",
        ):
            st.session_state[baseline_match_count_key] = 10
            st.rerun()
    stat_keys = [item.key for item in BASKETBALL_REFERENCE_STAT_ORDER]
    display = neighbors[
        ["player_name", "team", "position", *stat_keys, "similarity_score", "distance"]
    ].copy()
    for item in BASKETBALL_REFERENCE_STAT_ORDER:
        if item.value_format == "percentage":
            display[item.key] = (display[item.key] * 100).round(1).map(lambda x: f"{x:.1f}%")
        elif item.value_format == "number":
            display[item.key] = display[item.key].round(1)
        else:
            display[item.key] = display[item.key].astype(int)
    display["similarity_score"] = display["similarity_score"].map(lambda x: f"{x:.1%}")
    display["distance"] = display["distance"].round(3)
    render_copyable_table(
        display.rename(
            columns={
                "player_name": "Player",
                "team": "Team",
                "position": "Pos",
                **{
                    item.key: item.abbreviation
                    for item in BASKETBALL_REFERENCE_STAT_ORDER
                },
                "similarity_score": "Similarity",
                "distance": "Distance",
            }
        ),
    )
    st.caption(
        "Players are rows and statistics run left-to-right in Basketball Reference order. "
        "Similarity and distance are model outputs, so they appear last."
    )
    _advanced_retrieval_settings(style_available)

with why_tab:
    match_labels = {
        int(row.player_id): f"{row.player_name} · {row.team} · {row.position}"
        for row in neighbors.itertuples()
    }
    explained_id = st.selectbox(
        "Explain match",
        list(match_labels),
        format_func=match_labels.get,
        key=f"explain_match_{player_id}_{preset}",
    )
    match = neighbors.loc[neighbors["player_id"].eq(explained_id)].iloc[0]
    contribution = contribution_table(match)
    left, right = st.columns([1.05, 1])
    with left:
        radar_features = contribution.head(12)
        st.plotly_chart(
            similarity_radar_chart(
                radar_features,
                reference["player_name"],
                match["player_name"],
            ),
            width="stretch",
        )
        if len(contribution) > len(radar_features):
            st.caption("Radar shows the 12 largest distance contributions; the table includes every active feature.")
    with right:
        st.markdown(
            f"**{match['player_name']} is {match['similarity_score']:.1%} similar** "
            f"under the **{preset}** lens. Smaller percentile gaps are closer; "
            "distance contribution shows which differences reduced the score most."
        )
        formatted = contribution.copy()
        for column in ("Reference percentile", "Match percentile", "Percentile gap"):
            formatted[column] = formatted[column].map(lambda value: f"{value:.0%}")
        formatted["Distance contribution"] = formatted["Distance contribution"].map(
            lambda value: f"{value:.1%}"
        )
        render_copyable_table(formatted)

with ai_tab:
    st.write(
        "The retrieval model fixes the ranking. This deterministic template summarizes only "
        "the displayed match scores, feature gaps, contribution values, and limitations."
    )
    result_key = (
        f"similarity_result_{player_id}_{preset}_{top_n}_{position_aware}_{int(minimum_games)}_{int(minimum_shots)}"
    )
    if st.button("Explain these matches", type="primary"):
        with st.spinner("Explaining the fixed retrieval result…"):
            st.session_state[result_key] = generate_similarity_explanation(packet)
    if result_key in st.session_state:
        result = st.session_state[result_key]
        st.markdown(result.text)

with trace_tab:
    render_build_trace(
        [
            ("SQL cohort", f"Load the latest season rows and require at least {int(minimum_games)} games and {int(minimum_shots)} shot events."),
            ("Feature engineering", "Combine production, efficiency, creation, defense, shot selection, action type, distance, and expected-shot features."),
            ("Player embedding", "Convert every feature to a 0–1 percentile so unlike units can share one vector space."),
            ("Role weighting", f"Apply the transparent {preset} weight configuration."),
            ("Vector retrieval", "Calculate weighted Euclidean distance and return the nearest player embeddings."),
            ("Rule-based summary", "Translate the fixed matches and feature contributions into a cited deterministic explanation."),
        ],
        expanded=True,
    )
    active_families = {
        family: [feature for feature in features if feature in active_features]
        for family, features in FEATURE_FAMILIES.items()
    }
    active_families = {family: features for family, features in active_families.items() if features}
    st.markdown(f"**Current embedding: {len(active_features)} active features**")
    st.json(active_families, expanded=False)
    if shot_features_active:
        st.success(
            "NBA shot-detail features are active: zones, action types, shot distance, "
            "a smoothed expected-FG difficulty proxy, and shot-making above expected."
        )
    st.info(
        "Next data layers: playmaking/turnover context, on/off and lineup impact, "
        "defensive matchups, and physical role context."
    )
