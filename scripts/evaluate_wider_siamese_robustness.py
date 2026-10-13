"""Run the promotion gate for the wider Siamese Player DNA challenger."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ball_ai.analytics.play_style import MODERN_MOVE_FEATURES  # noqa: E402
from ball_ai.config import settings  # noqa: E402
from scripts.evaluate_broad_v2_robustness import (  # noqa: E402
    BOOTSTRAP_SEASON,
    SPLIT_SEASONS,
    _cross_validated_knn,
    _eligible,
    _feature_rows,
    _load_games,
    _load_shots,
    _split_season,
    neighbor_overlap,
)
from scripts.evaluate_siamese_robustness import HELD_OUT_TARGETS  # noqa: E402
from scripts.train_siamese_multiview_challenger import (  # noqa: E402
    DETAIL_WEIGHT,
    DETAILED_FEATURES,
    STABLE_FEATURES,
    _scores,
    _views,
    fit_challenger,
    transform,
)
from scripts.train_siamese_style_encoder import (  # noqa: E402
    _coverage_aware_scores,
)
from scripts.train_temporal_contrastive_encoder import _cohorts, _metrics  # noqa: E402


SEEDS = [7, 23, 42, 73, 101]
HIDDEN = (128, 64)
REPRESENTATIVE_PLAYERS = {
    "Shai Gilgeous-Alexander": "2025-26",
    "Stephen Curry": "2024-25",
    "Nikola Jokic": "2024-25",
    "LeBron James": "2018-19",
    "Shaquille O'Neal": "2001-02",
    "JJ Redick": "2009-10",
    "Luka Doncic": "2024-25",
    "Rudy Gobert": "2024-25",
}
CANDIDATE_DIR = (
    settings.historical_data_dir
    / "model_registry/play_style/candidates/siamese_wider_v1"
)


def _fit_pair(
    features: pd.DataFrame,
    views: pd.DataFrame,
    *,
    random_state: int,
    feature_names: list[str] | None = None,
) -> tuple[dict, dict]:
    stable_names = feature_names or STABLE_FEATURES
    detailed_names = [
        name for name in [*STABLE_FEATURES, *MODERN_MOVE_FEATURES]
        if feature_names is None or name in feature_names or name in MODERN_MOVE_FEATURES
    ]
    common = dict(
        features=features,
        views=views,
        training_end="2023-24",
        hidden=HIDDEN,
        use_views=False,
        reliability_weighted=False,
        random_state=random_state,
    )
    stable = fit_challenger(feature_names=stable_names, **common)
    detailed = fit_challenger(
        feature_names=detailed_names, training_start="2007-08", **common
    )
    return stable, detailed


def _new_scores(
    stable: dict, detailed: dict, query: pd.DataFrame, candidates: pd.DataFrame
) -> np.ndarray:
    stable_score = _scores(stable, query, candidates)
    detailed_score = _scores(detailed, query, candidates)
    coverage = (
        query["season"].ge("2007-08").to_numpy()[:, None]
        & candidates["season"].ge("2007-08").to_numpy()[None, :]
    )
    return np.where(
        coverage,
        (1 - DETAIL_WEIGHT) * stable_score + DETAIL_WEIGHT * detailed_score,
        stable_score,
    )


def _combined_vectors(
    stable: dict, detailed: dict, frame: pd.DataFrame
) -> np.ndarray:
    return np.concatenate([
        np.sqrt(1 - DETAIL_WEIGHT) * transform(stable, frame),
        np.sqrt(DETAIL_WEIGHT) * transform(detailed, frame),
    ], axis=1)


def _seed_stability(
    features: pd.DataFrame, views: pd.DataFrame
) -> tuple[list[dict], dict]:
    query, candidates = _cohorts(features, "2025-26", "2024-25")
    rows = []
    for seed in SEEDS:
        stable, detailed = _fit_pair(features, views, random_state=seed)
        rows.append({"seed": seed, **_metrics(
            _new_scores(stable, detailed, query, candidates), query, candidates
        )})
        print(f"Seed {seed}", flush=True)
    old = pd.DataFrame(json.loads(
        (ROOT / "models/play_style/siamese_tabular_v4_seed_stability.json").read_text()
    ))
    old = old.loc[old["version"].eq("v4")]
    current = pd.DataFrame(rows)
    metrics = ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
    return rows, {
        "wider_mean": current[metrics].mean().to_dict(),
        "frozen_v4_mean": old[metrics].mean().to_dict(),
        "wider_seed_standard_deviation": current[metrics].std(ddof=1).to_dict(),
    }


def _split_season_gate(stable: dict, detailed: dict) -> tuple[list[dict], dict]:
    rows = []
    for season in SPLIT_SEASONS:
        first, second = _split_season(settings.historical_data_dir, season)
        shared = set(first["player_id"].astype(int)) & set(second["player_id"].astype(int))
        first = first.loc[first["player_id"].astype(int).isin(shared)].reset_index(drop=True)
        second = second.loc[second["player_id"].astype(int).isin(shared)].reset_index(drop=True)
        rows.append({"season": season, **_metrics(
            _new_scores(stable, detailed, first, second), first, second
        )})
        print(f"Split season {season}", flush=True)
    current = pd.DataFrame(rows)
    old_payload = json.loads(
        (ROOT / "models/play_style/siamese_tabular_v4_robustness.json").read_text()
    )
    old = pd.DataFrame(old_payload["split_season"])
    old = old.loc[old["model"].eq("Coverage-aware Siamese")]
    metrics = ["mean_reciprocal_rank", "recall_at_1", "recall_at_5"]
    return rows, {
        "wider_mean": current[metrics].mean().to_dict(),
        "frozen_v4_mean": old[metrics].mean().to_dict(),
    }


def _bootstrap_gate(stable: dict, detailed: dict) -> dict:
    games = _load_games(settings.historical_data_dir, BOOTSTRAP_SEASON)
    shots = _load_shots(settings.historical_data_dir, BOOTSTRAP_SEASON)
    full = _eligible(_feature_rows(games, shots, BOOTSTRAP_SEASON))
    rows = []
    for repeat in range(10):
        sampled_games = games.groupby("player_id", group_keys=False).sample(
            frac=1, replace=True, random_state=42 + repeat
        )
        sampled = _eligible(_feature_rows(sampled_games, shots, BOOTSTRAP_SEASON))
        common = np.intersect1d(full["player_id"], sampled["player_id"])
        base = full.set_index("player_id").loc[common].reset_index()
        boot = sampled.set_index("player_id").loc[common].reset_index()
        left, right = (
            _combined_vectors(stable, detailed, base),
            _combined_vectors(stable, detailed, boot),
        )
        overlap = neighbor_overlap(left, right)
        rows.extend({
            "embedding_cosine": float(cosine),
            "top_10_neighbor_jaccard": float(jaccard),
        } for cosine, jaccard in zip(np.sum(left * right, axis=1), overlap))
        print(f"Bootstrap {repeat + 1}/10", flush=True)
    frame = pd.DataFrame(rows)
    old_payload = json.loads(
        (ROOT / "models/play_style/siamese_tabular_v4_robustness.json").read_text()
    )
    old = next(
        row for row in old_payload["bootstrap"]
        if row["model"] == "Coverage-aware Siamese"
    )
    return {
        "wider": {
            "mean_embedding_cosine": float(frame["embedding_cosine"].mean()),
            "p10_embedding_cosine": float(frame["embedding_cosine"].quantile(.10)),
            "mean_top_10_neighbor_jaccard": float(frame["top_10_neighbor_jaccard"].mean()),
            "p10_top_10_neighbor_jaccard": float(frame["top_10_neighbor_jaccard"].quantile(.10)),
        },
        "frozen_v4": old,
    }


def _shortcut_gate(
    features: pd.DataFrame, stable: dict, detailed: dict
) -> dict:
    query, candidates = _cohorts(features, "2025-26", "2024-25")
    scores = _new_scores(stable, detailed, query, candidates)
    candidate_ids = candidates["player_id"].astype(int).to_numpy()

    def ranks(ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.asarray([
            np.flatnonzero(ids[np.argsort(-scores[index])] == int(player_id))[0] + 1
            for index, player_id in enumerate(query["player_id"])
            if mask[index]
        ])

    training_ids = set(features.loc[
        features["offensive_eligible"].fillna(False)
        & features["season"].le("2023-24"), "player_id"
    ].astype(int))
    seen = query["player_id"].astype(int).isin(training_ids).to_numpy()
    groups = {}
    for label, mask in (("seen", seen), ("unseen", ~seen)):
        values = ranks(candidate_ids, mask)
        groups[label] = {
            "eligible_queries": len(values),
            "mean_reciprocal_rank": float(np.mean(1 / values)),
            "recall_at_1": float(np.mean(values == 1)),
            "recall_at_5": float(np.mean(values <= 5)),
        }
    probe = query.head(50).copy()
    before = np.concatenate([transform(stable, probe), transform(detailed, probe)], axis=1)
    for column, value in {
        "player_id": -999, "player_name": "IDENTITY LEAK", "team": "XXX",
        "position": "C", "age": 99, "height": 999, "weight": 999,
        "dpm": 999, "o_dpm": 999, "d_dpm": 999,
    }.items():
        probe[column] = value
    after = np.concatenate([transform(stable, probe), transform(detailed, probe)], axis=1)
    forbidden = {
        "player_id", "player_name", "team", "position", "height", "weight",
        "age", "dpm", "o_dpm", "d_dpm", "field_goal_percentage",
        "three_point_percentage", "effective_field_goal_percentage",
    }
    rng = np.random.default_rng(73)
    null = [
        float(np.mean(1 / ranks(rng.permutation(candidate_ids), np.ones(len(query), bool))))
        for _ in range(100)
    ]
    old = json.loads(
        (ROOT / "models/play_style/siamese_tabular_v4_robustness.json").read_text()
    )["shortcut_audit"]
    return {
        "forbidden_input_features": sorted(forbidden & (
            set(stable["feature_names"]) | set(detailed["feature_names"])
        )),
        "metadata_perturbation_max_abs_delta": float(np.max(np.abs(before - after))),
        "latest_test_by_training_exposure": groups,
        "permuted_identity_null_mrr_mean": float(np.mean(null)),
        "frozen_v4_latest_test_by_training_exposure": old["latest_test_by_training_exposure"],
    }


def _held_out_behavior(
    features: pd.DataFrame, views: pd.DataFrame
) -> tuple[list[dict], dict]:
    rows = []
    mask = features["offensive_eligible"].fillna(False) & features["season"].eq("2024-25")
    for target, excluded in HELD_OUT_TARGETS.items():
        stable_names = [name for name in STABLE_FEATURES if name not in excluded]
        detailed_names = [
            name for name in DETAILED_FEATURES if name not in excluded
        ]
        common = dict(
            features=features, views=views, training_end="2023-24", hidden=HIDDEN,
            use_views=False, reliability_weighted=False, random_state=42,
        )
        stable = fit_challenger(feature_names=stable_names, **common)
        detailed = fit_challenger(
            feature_names=detailed_names, training_start="2007-08", **common
        )
        cohort = features.loc[mask & features[target].notna()].reset_index(drop=True)
        actual = cohort[target].to_numpy(dtype=float)
        predicted, mean_predicted = _cross_validated_knn(
            _combined_vectors(stable, detailed, cohort), actual
        )
        scale = float(np.std(actual)) or 1
        rows.append({
            "target": target,
            "eligible_rows": len(cohort),
            "mae": float(mean_absolute_error(actual, predicted)),
            "normalized_mae": float(mean_absolute_error(actual, predicted) / scale),
            "r2": float(r2_score(actual, predicted)),
            "mean_baseline_mae": float(mean_absolute_error(actual, mean_predicted)),
        })
        print(f"Held-out behavior {target}", flush=True)
    current = pd.DataFrame(rows).set_index("target")
    old_payload = json.loads(
        (ROOT / "models/play_style/siamese_tabular_v4_robustness.json").read_text()
    )
    old = pd.DataFrame(old_payload["held_out_behavior"])
    old = old.loc[old["model"].eq("Coverage-aware Siamese")].set_index("target")
    ratio = current["normalized_mae"] / old["normalized_mae"]
    return rows, {
        "targets_improved": int((ratio < 1).sum()),
        "targets_within_five_percent": int((ratio <= 1.05).sum()),
        "normalized_mae_ratio_by_target": ratio.to_dict(),
    }


def _qualitative_matches(
    features: pd.DataFrame,
    stable: dict,
    detailed: dict,
    old_stable: dict,
    old_detailed: dict,
) -> dict:
    eligible = features.loc[features["offensive_eligible"].fillna(False)].reset_index(drop=True)
    output = {}
    for name, season in REPRESENTATIVE_PLAYERS.items():
        reference = eligible.loc[
            eligible["player_name"].eq(name) & eligible["season"].eq(season)
        ]
        if reference.empty:
            output[name] = {"season": season, "error": "Reference unavailable"}
            continue
        player_id = int(reference.iloc[0]["player_id"])
        candidates = eligible.loc[~eligible["player_id"].astype(int).eq(player_id)].copy()
        new = _new_scores(stable, detailed, reference, candidates)[0]
        _, old = _coverage_aware_scores(
            old_stable, old_detailed, reference, candidates, DETAIL_WEIGHT
        )

        def top(scores: np.ndarray) -> list[dict]:
            ranked = candidates.assign(similarity_score=scores).sort_values(
                "similarity_score", ascending=False
            ).drop_duplicates("player_id").head(5)
            return ranked[["player_name", "season", "similarity_score"]].to_dict(
                orient="records"
            )

        output[name] = {"season": season, "frozen_v4": top(old[0]), "wider": top(new)}
    return output


def _calibration(
    features: pd.DataFrame,
    stable: dict,
    detailed: dict,
    old_stable: dict,
    old_detailed: dict,
) -> dict:
    query, candidates = _cohorts(features, "2025-26", "2024-25")
    new = _new_scores(stable, detailed, query, candidates)
    _, old = _coverage_aware_scores(
        old_stable, old_detailed, query, candidates, DETAIL_WEIGHT
    )
    mask = query["player_id"].to_numpy()[:, None] != candidates["player_id"].to_numpy()[None, :]

    def summarize(scores: np.ndarray) -> dict:
        values = scores[mask]
        nearest = np.where(mask, scores, -np.inf).max(axis=1)
        return {
            "different_player_quantiles": {
                str(q): float(np.quantile(values, q)) for q in (.5, .75, .9, .95, .99)
            },
            "nearest_other_quantiles": {
                str(q): float(np.quantile(nearest, q)) for q in (.1, .5, .9)
            },
            "shares_above_threshold": {
                str(t): float(np.mean(values >= t)) for t in (.8, .85, .9, .94)
            },
        }

    return {"frozen_v4": summarize(old), "wider": summarize(new)}


def main() -> int:
    torch.set_num_threads(1)
    root = settings.historical_data_dir
    features = pd.read_parquet(
        root / "model_registry/play_style/v1/player_style_features.parquet"
    )
    views = _views(features)
    seed_rows, seed_summary = _seed_stability(features, views)
    stable, detailed = _fit_pair(features, views, random_state=42)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(stable, CANDIDATE_DIR / "siamese_model.joblib")
    joblib.dump(detailed, CANDIDATE_DIR / "detailed_siamese_model.joblib")
    split_rows, split_summary = _split_season_gate(stable, detailed)
    bootstrap = _bootstrap_gate(stable, detailed)
    shortcut = _shortcut_gate(features, stable, detailed)
    held_out_rows, held_out_summary = _held_out_behavior(features, views)

    old_stable = joblib.load(root / "model_registry/play_style/v4/siamese_model.joblib")
    old_detailed = joblib.load(
        root / "model_registry/play_style/v4/detailed_siamese_model.joblib"
    )
    qualitative = _qualitative_matches(
        features, stable, detailed, old_stable, old_detailed
    )
    calibration = _calibration(
        features, stable, detailed, old_stable, old_detailed
    )
    gates = {
        "seed_mean_mrr_beats_v4": (
            seed_summary["wider_mean"]["mean_reciprocal_rank"]
            > seed_summary["frozen_v4_mean"]["mean_reciprocal_rank"]
        ),
        "seed_mean_recall_at_1_beats_v4": (
            seed_summary["wider_mean"]["recall_at_1"]
            > seed_summary["frozen_v4_mean"]["recall_at_1"]
        ),
        "split_mrr_within_one_point": (
            split_summary["wider_mean"]["mean_reciprocal_rank"]
            >= split_summary["frozen_v4_mean"]["mean_reciprocal_rank"] - .01
        ),
        "bootstrap_cosine_within_one_point": (
            bootstrap["wider"]["mean_embedding_cosine"]
            >= bootstrap["frozen_v4"]["mean_embedding_cosine"] - .01
        ),
        "unseen_mrr_within_one_point": (
            shortcut["latest_test_by_training_exposure"]["unseen"]["mean_reciprocal_rank"]
            >= shortcut["frozen_v4_latest_test_by_training_exposure"]["unseen"]["mean_reciprocal_rank"] - .01
        ),
        "no_forbidden_metadata": not shortcut["forbidden_input_features"],
        "metadata_perturbation_is_zero": shortcut["metadata_perturbation_max_abs_delta"] == 0,
        "held_out_behavior_at_least_four_of_five_within_five_percent": (
            held_out_summary["targets_within_five_percent"] >= 4
        ),
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": "siamese-wider-v1-promotion-gate",
        "production_changed": False,
        "architecture": "shared 128 -> 64 MLP with 32-dimensional normalized embeddings",
        "automatic_gates": gates,
        "automatic_gate_pass": all(gates.values()),
        "promotion_decision": "eligible_for_review" if all(gates.values()) else "not_promoted",
        "qualitative_review_required": True,
        "seed_stability": seed_rows,
        "seed_summary": seed_summary,
        "split_season": split_rows,
        "split_season_summary": split_summary,
        "bootstrap": bootstrap,
        "shortcut_audit": shortcut,
        "held_out_behavior": held_out_rows,
        "held_out_behavior_summary": held_out_summary,
        "score_calibration": calibration,
        "representative_player_checks": qualitative,
    }
    tracked = ROOT / "models/play_style/siamese_wider_v1_robustness.json"
    tracked.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nAutomatic gates\n", json.dumps(gates, indent=2))
    print("\nSeed summary\n", json.dumps(seed_summary, indent=2))
    print("\nSplit-season summary\n", json.dumps(split_summary, indent=2))
    print("\nBootstrap\n", json.dumps(bootstrap, indent=2))
    print("\nShortcut audit\n", json.dumps(shortcut, indent=2))
    print("\nHeld-out behavior\n", json.dumps(held_out_summary, indent=2))
    for name, values in qualitative.items():
        print(f"\n{name} · {values.get('season')}")
        print("Frozen:", [row["player_name"] for row in values.get("frozen_v4", [])])
        print("Wider:", [row["player_name"] for row in values.get("wider", [])])
    print(f"Saved {tracked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
