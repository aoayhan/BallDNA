"""Compare frozen v4 and the wider encoder across five seeds and 19 folds."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ball_ai.config import settings  # noqa: E402
from scripts.train_siamese_multiview_challenger import (  # noqa: E402
    DETAIL_WEIGHT,
    DETAILED_FEATURES,
    STABLE_FEATURES,
    _scores as wider_component_scores,
    _views,
    fit_challenger,
)
from scripts.train_siamese_style_encoder import (  # noqa: E402
    _coverage_aware_scores,
    fit_siamese_encoder,
)
from scripts.train_temporal_contrastive_encoder import (  # noqa: E402
    ADDITIONAL_HOLDOUT_FOLDS,
    HOLDOUT_FOLDS,
    _cohorts,
    _metrics,
)


SEEDS = [7, 23, 42, 73, 101]
FOLDS = [*HOLDOUT_FOLDS, *ADDITIONAL_HOLDOUT_FOLDS]
WIDER_HIDDEN = (128, 64)


def _wider_scores(
    features: pd.DataFrame,
    views: pd.DataFrame,
    training_end: str,
    query: pd.DataFrame,
    candidates: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    common = dict(
        features=features,
        views=views,
        training_end=training_end,
        hidden=WIDER_HIDDEN,
        use_views=False,
        reliability_weighted=False,
        random_state=seed,
    )
    stable = fit_challenger(feature_names=STABLE_FEATURES, **common)
    stable_scores = wider_component_scores(stable, query, candidates)
    if training_end < "2008-09":
        return stable_scores
    detailed = fit_challenger(
        feature_names=DETAILED_FEATURES, training_start="2007-08", **common
    )
    detailed_scores = wider_component_scores(detailed, query, candidates)
    coverage = (
        query["season"].ge("2007-08").to_numpy()[:, None]
        & candidates["season"].ge("2007-08").to_numpy()[None, :]
    )
    return np.where(
        coverage,
        (1 - DETAIL_WEIGHT) * stable_scores + DETAIL_WEIGHT * detailed_scores,
        stable_scores,
    )


def _v4_scores(
    features: pd.DataFrame,
    training_end: str,
    query: pd.DataFrame,
    candidates: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    stable = fit_siamese_encoder(
        features,
        training_end=training_end,
        embedding_dimensions=32,
        temperature=.07,
        random_state=seed,
    )
    detailed = None
    if training_end >= "2008-09":
        detailed = fit_siamese_encoder(
            features,
            training_start="2007-08",
            training_end=training_end,
            embedding_dimensions=32,
            temperature=.07,
            feature_names=DETAILED_FEATURES,
            random_state=seed,
        )
    _, scores = _coverage_aware_scores(
        stable, detailed, query, candidates, DETAIL_WEIGHT
    )
    return scores


def _paired_summary(rows: pd.DataFrame) -> tuple[dict, list[dict], list[dict]]:
    metrics = ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
    summary = rows.groupby("model")[metrics].mean().to_dict(orient="index")
    pivot = rows.pivot(index=["seed", "fold"], columns="model", values=metrics)
    deltas = pd.DataFrame({
        metric: pivot[metric]["Wider"] - pivot[metric]["Frozen v4"]
        for metric in metrics
    }).reset_index()
    by_fold = deltas.groupby("fold", as_index=False)[metrics].mean()
    by_seed = deltas.groupby("seed", as_index=False)[metrics].mean()
    rng = np.random.default_rng(42)
    intervals = {}
    for metric in metrics:
        values = by_fold[metric].to_numpy()
        samples = np.asarray([
            rng.choice(values, len(values), replace=True).mean() for _ in range(20_000)
        ])
        intervals[metric] = {
            "mean_delta": float(values.mean()),
            "fold_wins": int((values > 0).sum()),
            "fold_ties": int((values == 0).sum()),
            "fold_losses": int((values < 0).sum()),
            "fold_bootstrap_95_percent_interval": [
                float(value) for value in np.quantile(samples, [.025, .975])
            ],
        }
    return {
        "model_means": summary,
        "paired_fold_evidence": intervals,
    }, by_fold.to_dict(orient="records"), by_seed.to_dict(orient="records")


def main() -> int:
    torch.set_num_threads(1)
    features = pd.read_parquet(
        settings.historical_data_dir
        / "model_registry/play_style/v1/player_style_features.parquet"
    )
    views = _views(features)
    rows = []
    for seed in SEEDS:
        for training_end, query_season, candidate_season in FOLDS:
            query, candidates = _cohorts(features, query_season, candidate_season)
            fold = f"{query_season} -> {candidate_season}"
            for model, scores in (
                ("Frozen v4", _v4_scores(
                    features, training_end, query, candidates, seed
                )),
                ("Wider", _wider_scores(
                    features, views, training_end, query, candidates, seed
                )),
            ):
                rows.append({
                    "seed": seed,
                    "fold": fold,
                    "model": model,
                    **_metrics(scores, query, candidates),
                })
            print(f"Seed {seed} · {fold}", flush=True)
    evaluations = pd.DataFrame(rows)
    summary, by_fold, by_seed = _paired_summary(evaluations)
    means = summary["model_means"]
    gates = {
        "mean_recall_at_1_beats_v4": (
            means["Wider"]["recall_at_1"] > means["Frozen v4"]["recall_at_1"]
        ),
        "mean_mrr_beats_v4": (
            means["Wider"]["mean_reciprocal_rank"]
            > means["Frozen v4"]["mean_reciprocal_rank"]
        ),
        "mean_recall_at_5_within_half_point": (
            means["Wider"]["recall_at_5"] >= means["Frozen v4"]["recall_at_5"] - .005
        ),
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": "siamese-wider-v1-five-seed-19-fold",
        "production_changed": False,
        "seeds": SEEDS,
        "fold_count": len(FOLDS),
        "promotion_gates": gates,
        "promotion_gate_pass": all(gates.values()),
        "summary": summary,
        "paired_deltas_by_fold": by_fold,
        "paired_deltas_by_seed": by_seed,
        "evaluations": evaluations.to_dict(orient="records"),
    }
    output = ROOT / "models/play_style/siamese_wider_v1_multiseed_folds.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nPromotion gates\n", json.dumps(gates, indent=2))
    print("\nModel means\n", json.dumps(means, indent=2))
    print("\nPaired fold evidence\n", json.dumps(
        summary["paired_fold_evidence"], indent=2
    ))
    print("\nPaired seed deltas\n", pd.DataFrame(by_seed).to_string(index=False))
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
