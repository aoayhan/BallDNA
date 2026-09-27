"""Recruiter-readable account of the deployed Player DNA product."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from _bootstrap import initialize_app  # noqa: E402
from components.styles import apply_styles  # noqa: E402
from components.ui import render_copyable_table  # noqa: E402
from ball_ai.config import settings  # noqa: E402
from ball_ai.data.database import get_data_metadata  # noqa: E402
from ball_ai.data.historical_store import (  # noqa: E402
    get_historical_coverage,
    get_historical_metadata,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


st.set_page_config(page_title="About · BallDNA", page_icon="🧬", layout="wide")
apply_styles()
initialize_app()
st.title("About This Project")
st.caption(
    "BallDNA is a self-supervised NBA play-style retrieval product I built to explore how "
    "AI and machine learning can be applied to basketball statistics, while designing new "
    "features, running experiments, and building interesting analytics products."
)

st.subheader("What is finished")
first, second, third = st.columns(3)
first.success("**Player Similarity · flagship**")
first.write(
    "Select a player-season and retrieve offensive, defensive(still working on), or overall historical style matches."
)
second.success("**Direct style comparison**")
second.write(
    "Compare any two available players and inspect their similarity score, shared tendencies, "
    "largest differences, efficiency, and impact difference."
)
third.success("**Evidence-first interface**")
third.write(
    "Every result exposes observed inputs, sample reliability, model configuration, and limitations."
)
st.warning(
    "🚧 Ask the Data, Team Needs Lab, and Roster Construction Lab remain under construction. "
    "They are supporting experiments, not finished products yet."
)

st.subheader("How Player DNA works")
st.code(
    """Historical box scores + shot events + matchup events
                         │
            validate coverage and preserve nulls
                         ▼
       player-season behavioral feature vectors
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       Broad History           Modern Detailed
       1996-97 onward          richer recent actions
             └───────────┬───────────┘
                         ▼
       denoising + temporal Player DNA
          + Siamese contrastive encoder
                         ▼
        cosine retrieval in learned space
                         ▼
 similarity score + tendencies + differences
                         │
                         ▼
       DARKO impact shown separately""",
    language=None,
)

left, right = st.columns(2)
with left:
    st.markdown("**What the model receives**")
    st.markdown(
        "- Shot zones, distance, and shot-type frequencies\n"
        "- Shooting volume, free-throw pressure, and estimated usage\n"
        "- Assists, turnovers, offensive rebounds, and ball-security behavior\n"
        "- Matchup and defensive event features where source coverage supports them\n"
        "- Modern action context such as assisted, fast-break, and second-chance shots"
    )
with right:
    st.markdown("**What is deliberately excluded**")
    st.markdown(
        "- Player identity and name\n"
        "- Team\n"
        "- Position\n"
        "- Height and weight\n"
        "- Shooting efficiency\n"
        "- DARKO impact"
    )
    st.caption(
        "Those exclusions prevent obvious identity and body-type shortcuts. Impact and efficiency "
        "remain visible as context, but playing alike does not mean being equally good."
    )

siamese = _read_json(
    settings.root_dir / "models/play_style/siamese_tabular_v3_summary.json"
)
if siamese:
    detail_weight = siamese.get("selected_detail_weight", 0.75)
    st.subheader("Deployed Broad History offensive ensemble")
    weight_columns = st.columns(2)
    weight_columns[0].metric("Stable Siamese", f"{1 - detail_weight:.0%}")
    weight_columns[1].metric("Detailed Siamese", f"{detail_weight:.0%}")
    st.caption(
        "Both shared MLPs learn from adjacent same-player seasons with contrastive loss. Detailed "
        "similarity is used only when both seasons have 2007-08+ move-label coverage; otherwise the "
        "score is 100% stable Siamese."
    )

if siamese.get("all_19_holdouts"):
    best = siamese["all_19_holdouts"]["coverage_aware_ensemble"]
    st.subheader("How it was evaluated")
    metrics = st.columns(3)
    metrics[0].metric("19-fold holdout MRR", f"{best['mean_reciprocal_rank']:.3f}")
    metrics[1].metric("Previous-season Top-1", f"{best['recall_at_1']:.1%}")
    metrics[2].metric("Previous-season Top-5", f"{best['recall_at_5']:.1%}")
    st.write(
        "Evaluation uses chronological holdouts: a player-season queries an earlier season pool, "
        "and success means retrieving that same player's adjacent season near the top. This tests "
        "temporal consistency without requiring subjective similarity labels. Cosine and PCA are "
        "retained as simpler baselines."
    )
    st.caption(
        f"The stable encoder achieved {siamese['all_19_holdouts']['stable_siamese']['mean_reciprocal_rank']:.3f} "
        f"MRR before coverage-aware detail improved it to {best['mean_reciprocal_rank']:.3f}. "
        "Split-season, bootstrap-stability, unseen-player, metadata-invariance, and permutation-null "
        "checks were run before promotion."
    )

metadata = get_data_metadata()
coverage = get_historical_coverage()
archive = get_historical_metadata()
st.subheader("Data pipeline and coverage")
if not coverage.empty:
    player_start = coverage.loc[coverage["box_score_rows"].notna(), "season"].min()
    team_start = coverage.loc[coverage["team_box_score_rows"].notna(), "season"].min()
    shot_start = coverage.loc[coverage["shot_event_rows"].notna(), "season"].min()
    latest = coverage["season"].max()
    coverage_columns = st.columns(3)
    coverage_columns[0].metric("Player box scores", f"{player_start} → {latest}")
    coverage_columns[1].metric("Team box scores", f"{team_start} → {latest}")
    coverage_columns[2].metric("Shot detail", f"{shot_start} → {latest}")
    counts = archive.get("row_counts", {})
    st.caption(
        f"{int(counts.get('box_score_rows', 0)):,} player rows · "
        f"{int(counts.get('team_box_score_rows', 0)):,} team rows · "
        f"{int(counts.get('shot_event_rows', 0)):,} shot events · Parquet with Snappy compression."
    )
st.write(
    "The pipeline normalizes source files into season-partitioned Parquet, records coverage by "
    "season and data type, and keeps unavailable historical fields null instead of converting them "
    "to zero. SQLite serves the compact current-season product views."
)
if metadata.get("provider_kind") == "kaggle_cc0_snapshot":
    st.caption(
        f"Current snapshot: {metadata.get('provider')} · {metadata.get('license')} · "
        f"dataset version {metadata.get('dataset_version')}. Historical shot detail comes from "
        "the local NBA Data Archive snapshot; DARKO DPM comes from the stored public nbarapm history snapshot."
    )

left, right = st.columns(2)
with left:
    st.subheader("What this demonstrates")
    st.markdown(
        "- **Machine learning:** denoising autoencoders, temporal metric learning, and a Siamese contrastive MLP\n"
        "- **Evaluation:** chronological holdouts, MRR, Recall@1/5, bootstrap stability, and baselines\n"
        "- **Data engineering:** multi-source ingestion, Parquet partitioning, SQLite, validation, and provenance\n"
        "- **Explainability:** observed feature gaps \n"
        "- **Product engineering:** searchable Streamlit workflows, interactive Plotly charts, caching, and tests\n"
    )
with right:
    st.subheader("Current limitations")
    st.markdown(
        "- Player similarity has no single objective ground truth because similarity depends on which aspects of play style are emphasized. Temporal self-retrieval provides a consistent evaluation proxy, but it does not define the one correct ranking of similar players.\n"
        "- **Defensive Player DNA is under development and should not be trusted by itself.**\n"
        "- Historical taxonomies and coverage are less detailed than modern data.\n"
        "- The system models structured behavior, not film, off-ball movement, injuries, chemistry, or coaching context.\n"
    )

st.info(
    "This project is not intended to predict the future or replace expert scouting. It demonstrates "
    "how structured basketball data, self-supervised learning, retrieval, and transparent evaluation "
    "can create explainable decision support."
)
