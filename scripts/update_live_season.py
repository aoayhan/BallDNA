"""Fetch the ongoing NBA season and publish provisional Player DNA assets."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ball_ai.analytics.play_style import build_embedding_frame  # noqa: E402
from ball_ai.analytics.siamese_attribution import (  # noqa: E402
    encode_siamese_features,
    load_siamese_inference_artifacts,
)
from ball_ai.data.live_season import (  # noqa: E402
    LIVE_MIN_GAMES,
    LIVE_MIN_SHOTS,
    build_live_offensive_features,
    current_nba_season,
)


LIVE = ROOT / "data" / "live"
HISTORY = ROOT / "data" / "historical"
SIAMESE_MODEL = ROOT / "models" / "play_style" / "siamese_tabular_v5_inference.npz"


def _request(factory, *, attempts: int = 3):
    """Retry a small number of transient NBA.com failures."""

    for attempt in range(attempts):
        try:
            return factory().get_data_frames()[0]
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(3 * (attempt + 1))


def fetch_nba_data(season: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch season totals, player metadata, and every completed shot."""

    try:
        from nba_api.stats.endpoints import (
            leaguedashplayerbiostats,
            leaguedashplayerstats,
            shotchartdetail,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install the local updater dependencies with: pip install -r requirements-live.txt"
        ) from exc

    stats = _request(lambda: leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="Totals",
        timeout=120,
    ))
    bios = _request(lambda: leaguedashplayerbiostats.LeagueDashPlayerBioStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_simple="Totals",
        timeout=120,
    ))
    shots = _request(lambda: shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=0,
        context_measure_simple="FGA",
        season_nullable=season,
        season_type_all_star="Regular Season",
        timeout=180,
    ))
    if stats.empty or shots.empty:
        raise RuntimeError(f"NBA.com returned no regular-season data for {season}.")
    return stats, bios, shots


def build_assets(
    stats: pd.DataFrame,
    bios: pd.DataFrame,
    shots: pd.DataFrame,
    *,
    season: str,
    minimum_games: int,
    minimum_shots: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the three compact files consumed by Streamlit."""

    historical_bios = pd.read_parquet(HISTORY / "players.parquet")
    features = build_live_offensive_features(
        stats,
        bios,
        shots,
        historical_bios,
        season=season,
        minimum_games=minimum_games,
        minimum_shots=minimum_shots,
    )
    eligible = features.loc[features["offensive_eligible"]].copy()
    if eligible.empty:
        raise RuntimeError("No current players passed the provisional live-season floor.")

    style_models = joblib.load(HISTORY / "play_style_models.joblib")
    artifact = style_models["Broad history"]["Offensive"]
    embeddings = build_embedding_frame(eligible, artifact, "Denoising autoencoder")
    historical_columns = pd.read_parquet(
        HISTORY / "player_style_embeddings.parquet"
    ).columns
    for column in historical_columns:
        if column.startswith("embedding_") and column not in embeddings:
            embeddings[column] = 0.0
    embeddings = embeddings.reindex(columns=historical_columns)

    siamese_artifacts = load_siamese_inference_artifacts(SIAMESE_MODEL)
    siamese = encode_siamese_features(eligible, siamese_artifacts)
    return features, embeddings, siamese


def _write_atomic(frame: pd.DataFrame, destination: Path, temporary: Path) -> None:
    path = temporary / destination.name
    frame.to_parquet(path, index=False, compression="snappy")
    path.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=current_nba_season())
    parser.add_argument("--minimum-games", type=int, default=LIVE_MIN_GAMES)
    parser.add_argument("--minimum-shots", type=int, default=LIVE_MIN_SHOTS)
    args = parser.parse_args()

    try:
        stats, bios, shots = fetch_nba_data(args.season)
        features, embeddings, siamese = build_assets(
            stats,
            bios,
            shots,
            season=args.season,
            minimum_games=args.minimum_games,
            minimum_shots=args.minimum_shots,
        )
        LIVE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="basketball_dna_live_") as temp:
            temporary = Path(temp)
            _write_atomic(features, LIVE / "player_style_features.parquet", temporary)
            _write_atomic(embeddings, LIVE / "player_style_embeddings.parquet", temporary)
            _write_atomic(siamese, LIVE / "siamese_offensive_embeddings.parquet", temporary)
        updated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        game_dates = pd.to_datetime(shots.get("GAME_DATE"), errors="coerce")
        metadata = {
            "provider": "NBA.com",
            "season": args.season,
            "updated_at": updated_at,
            "games_through": game_dates.max().date().isoformat() if game_dates.notna().any() else None,
            "provisional": True,
            "minimum_games": args.minimum_games,
            "minimum_shots": args.minimum_shots,
            "source_player_rows": int(len(stats)),
            "source_shot_rows": int(len(shots)),
            "eligible_players": int(embeddings["eligible"].sum()),
        }
        (LIVE / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    except Exception as exc:
        print(f"Live-season refresh failed safely; existing assets were preserved.\n{exc}")
        return 1

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
