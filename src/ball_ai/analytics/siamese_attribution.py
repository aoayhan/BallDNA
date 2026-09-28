"""Local, model-derived explanations for Siamese Player DNA similarity."""

from __future__ import annotations

from math import erf, pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from ball_ai.analytics.play_style import FEATURE_LABELS, SIAMESE_DETAIL_START


def load_siamese_inference_artifacts(path: Path) -> dict:
    """Load the small NumPy export of the deployed Siamese encoders."""

    with np.load(path, allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}

    def component(prefix: str) -> dict:
        return {
            "feature_names": data[f"{prefix}_feature_names"].astype(str).tolist(),
            "imputer_statistics": data[f"{prefix}_imputer_statistics"],
            "scaler_mean": data[f"{prefix}_scaler_mean"],
            "scaler_scale": data[f"{prefix}_scaler_scale"],
            "weights": [data[f"{prefix}_weight_{index}"] for index in range(3)],
            "biases": [data[f"{prefix}_bias_{index}"] for index in range(3)],
        }

    return {
        "stable": component("stable"),
        "detailed": component("detailed"),
        "detail_weight": float(data["detail_weight"].item()),
        "detail_start": str(data["detail_start"].item()),
    }


def _gelu(values: np.ndarray) -> np.ndarray:
    error = np.frompyfunc(erf, 1, 1)(values / sqrt(2)).astype(float)
    return 0.5 * values * (1 + error)


def _encode(component: dict, row: pd.Series, override: tuple[str, float] | None = None) -> np.ndarray:
    values = np.asarray([row.get(name, np.nan) for name in component["feature_names"]], dtype=float)
    if override is not None and override[0] in component["feature_names"]:
        values[component["feature_names"].index(override[0])] = override[1]
    values = np.where(np.isnan(values), component["imputer_statistics"], values)
    scale = np.where(component["scaler_scale"] == 0, 1, component["scaler_scale"])
    values = (values - component["scaler_mean"]) / scale
    for index, (weights, bias) in enumerate(zip(component["weights"], component["biases"])):
        values = values @ weights.T + bias
        if index < 2:
            values = _gelu(values)
    norm = np.linalg.norm(values)
    return values / norm if norm else values


def _score(
    component: dict,
    reference: pd.Series,
    match: pd.Series,
    reference_override: tuple[str, float] | None = None,
    match_override: tuple[str, float] | None = None,
) -> float:
    cosine = float(np.clip(
        _encode(component, reference, reference_override)
        @ _encode(component, match, match_override),
        -1,
        1,
    ))
    return 1 - np.arccos(cosine) / pi


def siamese_pair_attribution(
    features: pd.DataFrame,
    artifacts: dict,
    reference_id: int,
    match_id: int,
    reference_season: str,
    match_season: str,
) -> tuple[float, pd.DataFrame]:
    """Measure each input's local effect on one deployed Siamese pair score."""

    reference_rows = features.loc[
        features["player_id"].astype(int).eq(int(reference_id))
        & features["season"].eq(reference_season)
    ]
    match_rows = features.loc[
        features["player_id"].astype(int).eq(int(match_id))
        & features["season"].eq(match_season)
    ]
    if reference_rows.empty or match_rows.empty:
        raise ValueError("Both player-seasons need style features for model attribution.")
    reference = reference_rows.iloc[0]
    match = match_rows.iloc[0]
    components = [(artifacts["stable"], 1.0)]
    if min(reference_season, match_season) >= artifacts.get("detail_start", SIAMESE_DETAIL_START):
        detail_weight = float(artifacts["detail_weight"])
        components = [
            (artifacts["stable"], 1 - detail_weight),
            (artifacts["detailed"], detail_weight),
        ]

    baseline = sum(weight * _score(component, reference, match) for component, weight in components)
    feature_names = list(dict.fromkeys(
        name for component, _ in components for name in component["feature_names"]
    ))
    rows = []
    for feature in feature_names:
        similarity_contribution = 0.0
        difference_penalty = 0.0
        for component, weight in components:
            if feature not in component["feature_names"]:
                continue
            index = component["feature_names"].index(feature)
            median = float(component["imputer_statistics"][index])
            reference_value = pd.to_numeric(pd.Series([reference.get(feature)]), errors="coerce").iloc[0]
            match_value = pd.to_numeric(pd.Series([match.get(feature)]), errors="coerce").iloc[0]
            reference_value = median if pd.isna(reference_value) else float(reference_value)
            match_value = median if pd.isna(match_value) else float(match_value)
            component_baseline = _score(component, reference, match)
            neutralized = 0.5 * (
                _score(component, reference, match, reference_override=(feature, median))
                + _score(component, reference, match, match_override=(feature, median))
            )
            equalized = 0.5 * (
                _score(component, reference, match, reference_override=(feature, match_value))
                + _score(component, reference, match, match_override=(feature, reference_value))
            )
            similarity_contribution += weight * (component_baseline - neutralized)
            difference_penalty += weight * (equalized - component_baseline)
        rows.append({
            "Feature": FEATURE_LABELS.get(feature, feature),
            "Reference": reference.get(feature),
            "Match": match.get(feature),
            "Similarity contribution": similarity_contribution,
            "Difference penalty": difference_penalty,
        })
    return baseline, pd.DataFrame(rows)


def format_attribution_table(
    attribution: pd.DataFrame,
    metric: str,
    reference_name: str,
    match_name: str,
    limit: int = 8,
) -> pd.DataFrame:
    """Format the strongest positive local score effects for display."""

    display = attribution.loc[attribution[metric].gt(0), [
        "Feature", "Reference", "Match", metric,
    ]].nlargest(limit, metric).rename(columns={
        "Reference": reference_name,
        "Match": match_name,
        metric: "Model score effect",
    })
    for column in (reference_name, match_name):
        display[column] = pd.to_numeric(display[column], errors="coerce").map(
            lambda value: "—" if pd.isna(value) else f"{value:.3f}"
        )
    display["Model score effect"] = display["Model score effect"].map(
        lambda value: f"+{100 * value:.2f} pp"
    )
    return display
