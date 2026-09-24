"""Grounded reports, comparisons, and answers with a no-key fallback."""

from __future__ import annotations

from dataclasses import dataclass

from ball_ai.ai.llm_client import LLMUnavailableError, generate_text
from ball_ai.ai.prompts import (
    SYSTEM_INSTRUCTIONS,
    comparison_prompt,
    question_prompt,
    scouting_prompt,
    similarity_prompt,
    roster_simulation_prompt,
    team_needs_prompt,
)
from ball_ai.analytics.trends import trend_sentence


@dataclass(frozen=True)
class GenerationResult:
    text: str
    mode: str
    warning: str | None = None


def _format_value(item: dict) -> str:
    value = item["value"]
    if item["format"] == "percentage":
        return f"{value:.1%}"
    if item["format"] == "integer":
        return str(int(value))
    return f"{value:.1f}"


def _evidence_lookup(packet: dict) -> dict[str, dict]:
    return {item["metric"]: item for item in packet["evidence"]}


def _reliable_three_point_sample(stats: dict) -> bool:
    games = float(stats.get("games_played", 0) or 0)
    attempts = float(stats.get("three_point_attempts_per_game", 0) or 0) * games
    makes = float(stats.get("three_points_made_per_game", 0) or 0) * games
    return attempts >= 25 and makes >= 5


def fallback_scouting_report(packet: dict) -> str:
    """Create a deterministic, citation-rich report without an API call."""

    player = packet["player"]
    stats = packet["season_stats"]
    by_metric = _evidence_lookup(packet)
    scoring = by_metric["points_per_game"]
    playmaking = by_metric["assists_per_game"]
    rebounding = by_metric["rebounds_per_game"]
    three_point = by_metric.get("three_point_percentage")
    turnovers = by_metric["turnovers_per_game"]

    primary_position = str(player.get("position", "N/A")).split("-")[0]
    if primary_position == "G":
        primary_role = (
            "an on-ball creation role"
            if stats["assists_per_game"] >= 5
            else "a perimeter or secondary-guard role"
        )
    elif primary_position == "C":
        primary_role = "a frontcourt finishing or rebounding role"
    elif primary_position == "F":
        primary_role = "a wing or forward role"
    else:
        primary_role = "a role based on the supplied production profile"
    reliable_three = _reliable_three_point_sample(stats)
    if reliable_three and float(stats.get("three_point_percentage", 0)) >= 0.35:
        spacing_role = "lineups that value perimeter spacing"
    else:
        spacing_role = "lineups that do not depend on this player for proven spacing"

    strengths = []
    if float(scoring["value"]) >= 15:
        strengths.append(
            f"- Scoring volume: {_format_value(scoring)} points per game [{scoring['evidence_id']}]."
        )
    if float(playmaking["value"]) >= 5:
        strengths.append(
            f"- Playmaking volume: {_format_value(playmaking)} assists per game [{playmaking['evidence_id']}]."
        )
    if float(rebounding["value"]) >= 7:
        strengths.append(
            f"- Rebounding production: {_format_value(rebounding)} per game [{rebounding['evidence_id']}]."
        )
    if reliable_three and three_point and float(three_point["value"]) >= 0.35:
        strengths.append(
            f"- Supported perimeter shooting: {_format_value(three_point)} on at least 25 attempts and 5 makes [{three_point['evidence_id']}]."
        )
    if not strengths:
        strengths.append(
            f"- Role production: {_format_value(scoring)} points, {_format_value(rebounding)} rebounds, and {_format_value(playmaking)} assists per game [{scoring['evidence_id']}] [{rebounding['evidence_id']}] [{playmaking['evidence_id']}]."
        )
    role_citations = " ".join(
        f"[{item['evidence_id']}]"
        for item in (playmaking, rebounding, three_point)
        if item
    )

    shot_profile = packet.get("shot_profile", {})
    shot_strengths = ""
    shot_risk = "- The data does not measure defender distance, screening, injuries, or lineup context."
    if shot_profile:
        zone_keys = [
            "rim_frequency", "paint_frequency", "midrange_frequency",
            "three_point_frequency",
        ]
        action_keys = [
            "dunk_frequency", "layup_frequency", "floater_frequency",
            "hook_frequency", "pull_up_frequency", "step_back_frequency",
        ]
        dominant_zone = max(zone_keys, key=lambda key: float(shot_profile[key]))
        dominant_action = max(action_keys, key=lambda key: float(shot_profile[key]))
        zone_evidence = by_metric[dominant_zone]
        action_evidence = by_metric[dominant_action]
        modeled = by_metric["shot_making_above_expected"]
        shot_strengths = (
            f"\n- Shot selection: {_format_value(zone_evidence)} of attempts came from "
            f"{zone_evidence['label'].removesuffix(' attempt frequency').lower()} "
            f"[{zone_evidence['evidence_id']}].\n"
            f"- Shot type: {_format_value(action_evidence)} of attempts used the leading "
            f"action category, {action_evidence['label'].removesuffix(' attempt frequency').lower()} "
            f"[{action_evidence['evidence_id']}]."
        )
        direction = "above" if float(modeled["value"]) >= 0 else "below"
        shot_risk = (
            f"- Actual shot making was {abs(float(modeled['value'])):.1%} {direction} the "
            f"zone/action expectation [{modeled['evidence_id']}]. This model does not observe defender pressure."
        )

    trend_text = trend_sentence(packet["trends"])
    recent_label = (
        "stored licensed game rows"
        if str(packet.get("data_status", "")).startswith(
            "CC0-licensed Kaggle snapshot"
        )
        else "illustrative games"
    )
    trend_citations = " ".join(
        f"[{item['evidence_id']}]"
        for item in packet["evidence"]
        if item["metric"].startswith("trend_")
    )
    trend_section = (
        f"{trend_text} {trend_citations} These splits use "
        f"{packet['recent_summary'].get('games', 0)} {recent_label} and should be "
        "treated as directional, not definitive."
        if packet["trends"]
        else f"{trend_text} At least six games are required for two three-game windows."
    )
    evidence_lines = "\n".join(
        f"- [{item['evidence_id']}] {item['label']}: {_format_value(item)} ({item['context']})"
        for item in packet["evidence"]
    )
    limitations = " ".join(packet["limitations"])
    return f"""## Summary
{player['player_name']} is producing {_format_value(scoring)} points, {_format_value(rebounding)} rebounds, and {_format_value(playmaking)} assists per game [{scoring['evidence_id']}] [{rebounding['evidence_id']}] [{playmaking['evidence_id']}]. This is a data-only reading of the stored {player['season']} snapshot.

## Strengths
{chr(10).join(strengths)}
{shot_strengths}

## Weaknesses or risks
- Ball security is a monitoring point at {_format_value(turnovers)} turnovers per game [{turnovers['evidence_id']}]. The packet does not include usage or time-of-possession, so the cause cannot be isolated.
{shot_risk}

## Recent trend
{trend_section}

## Best role fit
The supplied position and production are most consistent with {primary_role} and {spacing_role} {role_citations}. This is a conditional statistical fit, not a film-based role assignment.

## Evidence used
{evidence_lines}

## Confidence and limitations
Confidence is moderate for describing the supplied statistical profile and low for real-world scouting conclusions. {limitations}
""".strip()


def fallback_comparison(packet: dict) -> str:
    """Create a deterministic two-player comparison from supplied values."""

    first, second = packet["players"]
    a, b = first["player"], second["player"]
    metrics = [
        ("three_point_percentage", "three-point shooting"),
        ("rebounds_per_game", "rebounding"),
        ("assists_per_game", "playmaking"),
        ("points_per_game", "scoring"),
    ]
    differences = []
    similarities = []
    for key, label in metrics:
        if key not in first["season_stats"] or key not in second["season_stats"]:
            continue
        av = float(first["season_stats"][key])
        bv = float(second["season_stats"][key])
        ae = _evidence_lookup(first)[key]
        be = _evidence_lookup(second)[key]
        threshold = .02 if "percentage" in key else 1.0
        line = (
            f"{a['player_name']}: {_format_value(ae)} [{a['player_name']} {ae['evidence_id']}]; "
            f"{b['player_name']}: {_format_value(be)} [{b['player_name']} {be['evidence_id']}]."
        )
        (similarities if abs(av - bv) <= threshold else differences).append(f"- {label.title()}: {line}")

    reliable_spacers = [
        player_packet
        for player_packet in (first, second)
        if _reliable_three_point_sample(player_packet["season_stats"])
        and "three_point_percentage" in player_packet["season_stats"]
    ]
    spacer_packet = (
        max(
            reliable_spacers,
            key=lambda item: item["season_stats"]["three_point_percentage"],
        )
        if reliable_spacers
        else None
    )
    creator_packet = first if first["season_stats"]["assists_per_game"] >= second["season_stats"]["assists_per_game"] else second
    creator = creator_packet["player"]
    creator_evidence = _evidence_lookup(creator_packet)["assists_per_game"]
    if spacer_packet:
        spacer = spacer_packet["player"]
        spacer_evidence = _evidence_lookup(spacer_packet)["three_point_percentage"]
        spacing_line = (
            f"- A spacing-oriented need points to {spacer['player_name']} among players with "
            f"at least 25 attempts and 5 makes [{spacer['player_name']} {spacer_evidence['evidence_id']}]."
        )
    else:
        spacing_line = "- Neither supplied player has a reliable three-point sample for a spacing recommendation."
    return f"""## Summary
{a['player_name']} and {b['player_name']} are compared as need-dependent statistical profiles, not as an overall ranking.

## Similarities
{chr(10).join(similarities) if similarities else '- No close match appeared within the simple display thresholds.'}

## Differences
{chr(10).join(differences)}

## Need-based fit
{spacing_line}
- A facilitation-oriented need points to {creator['player_name']} based on the higher supplied assists-per-game value [{creator['player_name']} {creator_evidence['evidence_id']}].

## Evidence used
All values and citations are listed in the Similarities and Differences sections above.

## Confidence and limitations
Confidence is moderate for the numerical comparison and low for team-fit conclusions. The demo omits lineup, role, salary, health, and film context.
""".strip()


def fallback_answer(packet: dict, question: str) -> str:
    """Answer common product questions using transparent deterministic rules."""

    name = packet["player"]["player_name"]
    lookup = _evidence_lookup(packet)
    lowered = question.lower()
    if any(word in lowered for word in ("underperform", "changed", "recent", "trend")):
        answer = trend_sentence(packet["trends"])
        citations = " ".join(
            f"[{item['evidence_id']}]"
            for item in packet["evidence"]
            if item["metric"].startswith("trend_")
        )
        return (
            f"**Direct answer:** For {name}, {answer} {citations}\n\n"
            f"This uses the stored recent-game split and the evidence shown below; it cannot identify causes such as injury, role, or opponent quality."
        )
    if any(word in lowered for word in ("facilitator", "playmaker", "scorer")):
        points = lookup["points_per_game"]
        assists = lookup["assists_per_game"]
        position = set(str(packet["player"].get("position", "N/A")).split("-"))
        if points["value"] >= 20 and assists["value"] >= 7:
            description = "a dual scoring-and-facilitation hub"
        elif points["value"] >= 18:
            description = "scoring-led"
        elif assists["value"] >= 5:
            description = "facilitation-led"
        elif "C" in position:
            description = "primarily a finishing/rebounding center, not a primary scorer or facilitator"
        else:
            description = "a secondary contributor rather than a primary scorer or facilitator"
        return (
            f"**Direct answer:** The supplied profile is {description}: {_format_value(points)} points per game "
            f"[{points['evidence_id']}] and {_format_value(assists)} assists per game [{assists['evidence_id']}].\n\n"
            "Limitation: possession share and passing-quality data are not available."
        )
    shot_profile = packet.get("shot_profile", {})
    if shot_profile and any(
        word in lowered
        for word in (
            "shot selection", "shot type", "difficulty", "floater", "pull-up",
            "pullup", "step-back", "step back", "midrange", "rim", "layup", "dunk",
        )
    ):
        zones = ["rim_frequency", "paint_frequency", "midrange_frequency", "three_point_frequency"]
        actions = [
            "dunk_frequency", "layup_frequency", "floater_frequency",
            "hook_frequency", "pull_up_frequency", "step_back_frequency",
        ]
        zone = lookup[max(zones, key=lambda key: float(shot_profile[key]))]
        action = lookup[max(actions, key=lambda key: float(shot_profile[key]))]
        difficulty = lookup["shot_difficulty_index"]
        return (
            f"**Direct answer:** {name}'s leading shot zone is {zone['label'].removesuffix(' attempt frequency').lower()} "
            f"at {_format_value(zone)} of attempts [{zone['evidence_id']}], while the most frequent "
            f"labeled action is {action['label'].removesuffix(' attempt frequency').lower()} at "
            f"{_format_value(action)} [{action['evidence_id']}]. The expected-shot difficulty "
            f"index is {_format_value(difficulty)} [{difficulty['evidence_id']}].\n\n"
            "Limitation: difficulty is inferred from zone/action make rates and does not directly observe defender pressure."
        )
    if any(word in lowered for word in ("spacing", "shoot", "three")):
        three = lookup.get("three_point_percentage")
        volume = lookup.get("three_point_frequency")
        if not three or not _reliable_three_point_sample(packet["season_stats"]):
            return (
                f"**Direct answer:** The supplied data does not contain a reliable three-point "
                f"sample for {name}; BallDNA requires at least 25 attempts and 5 makes.\n\n"
                "Defensive attention and off-ball gravity are not directly measured."
            )
        volume_text = (
            f" Three-point attempts represent {_format_value(volume)} of the shot profile "
            f"[{volume['evidence_id']}]."
            if volume
            else ""
        )
        return (
            f"**Direct answer:** The available spacing signal is {_format_value(three)} from three "
            f"[{three['evidence_id']}].{volume_text} Defensive attention and off-ball gravity are not directly measured."
        )
    return (
        f"The current evidence packet cannot answer that question reliably for {name}. "
        "Try asking about scoring, facilitation, spacing, or recent statistical trends.\n\n"
        "Limitation: the fallback mode intentionally refuses claims outside the supplied fields."
    )


def _similarity_value(item: dict, key: str) -> str:
    value = float(item[key])
    percentage_feature = (
        item["label"].endswith("%")
        or item["feature"].endswith("_frequency")
        or item["feature"] in {"shot_difficulty_index", "shot_making_above_expected"}
    )
    return f"{value:.1%}" if percentage_feature else f"{value:.1f}"


def fallback_similarity_explanation(packet: dict) -> str:
    """Explain deterministic similarity retrieval without an API key."""

    reference = packet["reference_player"]["player_name"]
    matches = packet.get("matches", [])[:3]
    if not matches:
        return "No matches were available to explain."

    summaries = []
    differences = []
    for match in matches:
        closest = sorted(
            match["feature_evidence"], key=lambda item: item["percentile_gap"]
        )[:2]
        furthest = max(
            match["feature_evidence"], key=lambda item: item["distance_contribution"]
        )
        close_text = "; ".join(
            f"{item['label']} {_similarity_value(item, 'reference_value')} vs "
            f"{_similarity_value(item, 'match_value')} [{item['evidence_id']}]"
            for item in closest
        )
        summaries.append(
            f"- **{match['player_name']}** ({match['similarity_score']:.1%} similarity): {close_text}."
        )
        differences.append(
            f"- **{match['player_name']}**: {furthest['label']} contributes the largest share "
            f"of this match's distance ({furthest['distance_contribution']:.1%}) "
            f"[{furthest['evidence_id']}]."
        )

    limitations = " ".join(packet["limitations"])
    method = packet["retrieval_method"]
    return f"""## Retrieval result
The deterministic {method['preset']} embedding retrieved the closest supplied profiles to {reference}. The language layer did not choose or reorder these players.

## Why the matches are similar
{chr(10).join(summaries)}

## Important differences
{chr(10).join(differences)}

## Confidence and limitations
Confidence is moderate for distance within the current feature space and low for complete play-style equivalence. {limitations}
""".strip()


def _team_need_value(feature: str, value: float) -> str:
    if any(
        token in feature
        for token in ("percentage", "_rate", "_share", "assist_ratio")
    ):
        return f"{float(value):.1%}"
    return f"{float(value):.1f}"


def fallback_team_needs_report(packet: dict) -> str:
    """Explain fixed team gaps and candidates without an API key."""

    team = packet["team"]
    common = packet.get("elite_commonalities", [])[:3]
    gaps = packet.get("team_gaps", [])[:4]
    players = packet.get("candidate_examples", [])[:3]
    rotation = packet.get("rotation_candidate_examples", [])[:2]
    common_lines = "\n".join(
        f"- {item['label']} consistently points {item['desired_direction']} among the "
        f"elite cohort ({item['elite_stability']:.0%} stability) [{item['evidence_id']}]."
        for item in common
    ) or "- No stable commonality passed the available-data checks."
    gap_lines = "\n".join(
        f"- {item['label']}: {team['team']} is at "
        f"{_team_need_value(item['feature'], item['team_value'])}, versus an elite median of "
        f"{_team_need_value(item['feature'], item['elite_median'])}; the model prefers "
        f"{item['desired_direction']} values [{item['evidence_id']}]."
        for item in gaps
    ) or "- The selected team has no positive model-weighted gaps in this benchmark."
    traits = []
    for gap in gaps:
        if gap["needed_player_trait"] not in traits:
            traits.append(gap["needed_player_trait"])
    archetype = ", ".join(traits[:3]) or "balanced contributor"
    candidate_lines = "\n".join(
        f"- **{item['player_name']} ({item['team']})** — {item['fit_score']:.1%} trait alignment, "
        f"{item.get('selection_rate', 1.0):.0%} sensitivity-grid robustness; "
        f"{item['fit_reasons']} [{item['evidence_id']}]."
        for item in players
    ) or "- No candidate passed the selected availability and sample filters."
    rotation_lines = "\n".join(
        f"- **{item['player_name']} ({item['team']})** — {item['fit_score']:.1%} trait alignment "
        f"[{item['evidence_id']}]."
        for item in rotation
    )
    limitations = " ".join(packet["limitations"])
    return f"""## Executive read
{team['team']} finished {int(team['league_rank'])}th in the supplied {team['season']} league-wide regular-season ranking. The analysis compares its measurable profile with the benchmark cohort; it does not forecast a future finish.

## What elite teams share
{common_lines}

## Priority gaps
{gap_lines}

## Needed player archetype
The highest-priority statistical needs translate to a **{archetype}** profile. This translation uses fixed mappings from team gaps to player traits [{gaps[0]['evidence_id'] if gaps else 'no-gap'}].

## Candidate examples
Best statistical matches:
{candidate_lines}

Rotation-scale examples:
{rotation_lines or '- No separate rotation-scale example passed the filters.'}

## Confidence and limitations
Confidence is moderate for describing the team profile and ranking candidates inside the supplied feature space, and low for causal roster conclusions. {limitations}
""".strip()


def fallback_roster_simulation_report(packet: dict) -> str:
    """Explain a fixed roster counterfactual without an API key."""

    estimates = {item["evidence_id"]: item for item in packet["quality_estimates"]}
    support = packet.get("model_support", {"is_supported": True})
    talent = packet.get("roster_talent", {})
    upside = packet.get("sandbox_upside", {})
    changes = packet.get("largest_embedding_changes", [])
    profiles = packet.get("simulated_team_profiles", [])
    neighbours = packet.get("nearest_historical_teams", [])
    moves = packet.get("roster_moves", {})
    move_parts = []
    if moves.get("added"):
        move_parts.append(f"adding {', '.join(moves['added'])}")
    if moves.get("removed"):
        move_parts.append(f"removing {', '.join(moves['removed'])}")
    move_description = " and ".join(move_parts) or "making no roster changes"
    detection = packet.get("change_detection", {})
    validation = packet.get("counterfactual_validation", {})
    resolution = float(detection.get("resolution_threshold", 0.05))
    if validation.get("is_counterfactual") and not validation.get("outcome_validated"):
        support_note = ""
        if not support.get("is_supported", True):
            ratio = support.get("distance_to_boundary_ratio")
            support_note = (
                f" It is also {float(ratio):.1f}× the historical support boundary away [S1]."
                if ratio is not None
                else " It is also outside historical model support [S1]."
            )
        percentile = talent.get("historical_percentile")
        talent_note = (
            f" Its {float(percentile):.0f}th-percentile talent index is descriptive, not an expected outcome [T1]."
            if percentile is not None
            else ""
        )
        quality_sentence = (
            "does not receive a validated expected Net Rating or win change because historical "
            f"transactions have not yet backtested this counterfactual model [V1].{support_note}{talent_note}"
        )
        range_sentence = ""
    elif not support.get("is_supported", True):
        ratio = support.get("distance_to_boundary_ratio")
        ratio_text = f"{float(ratio):.1f}×" if ratio is not None else "beyond"
        percentile = talent.get("historical_percentile")
        percentile_text = (
            f"{float(percentile):.0f}th-percentile"
            if percentile is not None
            else "unscored"
        )
        quality_sentence = (
            "does not receive an evidence-grade Net Rating or win estimate because it is "
            f"{ratio_text} the historical support boundary away [S1]. Its "
            f"{percentile_text} talent index is descriptive, not an outcome forecast [T1]"
        )
        range_sentence = ""
    elif abs(estimates["Q3"]["value"]) < resolution:
        quality_sentence = (
            f"shows no detectable change: the raw estimate moves from "
            f"{estimates['Q1']['value']:+.1f} to {estimates['Q2']['value']:+.1f}, "
            f"which is below the {resolution:.2f}-point model resolution [Q1] [Q2] [Q3]"
        )
        range_sentence = (
            f" The associated 80% empirical range is {estimates['Q4']['low']:.0f}–"
            f"{estimates['Q4']['high']:.0f} wins [Q4]."
        )
    else:
        direction = "improves" if estimates["Q3"]["value"] > 0 else "declines"
        quality_sentence = (
            f"{direction} from {estimates['Q1']['value']:+.1f} to "
            f"{estimates['Q2']['value']:+.1f} Net Rating [Q1] [Q2] [Q3]"
        )
        range_sentence = (
            f" The associated 80% empirical range is {estimates['Q4']['low']:.0f}–"
            f"{estimates['Q4']['high']:.0f} wins [Q4]."
        )
    change_lines = "\n".join(
        f"- **{item['trait']}** moves from {item['baseline_percentile']:.0%} to "
        f"{item['simulated_percentile']:.0%} ({item['change']:+.0%}) [{item['evidence_id']}]."
        for item in changes[:4]
    ) or "- No material trait changes were calculated."
    profile_lines = "\n".join(
        f"- **{item['profile']}**: {item['score']:.0%} profile score [{item['evidence_id']}]."
        for item in profiles[:3]
    ) or "- No profile scores were available."
    neighbour_lines = "\n".join(
        f"- **{item['team']} · {item['season']}**: {item['wins']}-{item['games']-item['wins']}, "
        f"{item['net_rating']:+.1f} Net Rating [{item['evidence_id']}]."
        for item in neighbours[:3]
    ) or "- No historical neighbours were available."
    weakest = list(reversed(profiles[-2:])) if profiles else []
    need_lines = "\n".join(
        f"- {item['profile']} is one of the lower supplied identity scores at "
        f"{item['score']:.0%} [{item['evidence_id']}]."
        for item in weakest
    ) or "- The packet does not identify a remaining profile need."
    limitations = " ".join(packet.get("limitations", []))
    added_names = set(packet.get("roster_moves", {}).get("added", []))
    added_rotation = [
        item
        for item in packet.get("normalized_rotation", [])
        if item.get("player_name") in added_names
    ]
    rotation_lines = "\n".join(
        f"- **{item['player_name']}** brings {item['observed_minutes_per_game']:.1f} source-team MPG "
        f"but earns {item['automatic_rotation_minutes']:.1f} MPG in this roster "
        f"[{item['evidence_id']}]."
        for item in added_rotation
    ) or "- No additions require a separate role-allocation note."
    upside_sentence = ""
    if upside.get("talent_ceiling_wins") is not None:
        upside_sentence = (
            f" The separate monotonic model gives an experimental point estimate of "
            f"{int(upside['talent_ceiling_wins'])} wins "
            f"({int(upside['talent_ceiling_wins'])}-{int(upside['talent_ceiling_losses'])}) "
            f"and {float(upside['talent_ceiling_net_rating']):+.1f} Net Rating for roster "
            "comparison; transaction outcomes have not yet calibrated it as a real forecast [U1]."
        )
    return f"""## Simulation read
After {move_description}, the roster profile {quality_sentence}{range_sentence}{upside_sentence}

## Identity changes
{change_lines}

Automatic role allocation:
{rotation_lines}

The strongest simulated identities are:
{profile_lines}

## Historical neighbours
{neighbour_lines}

## What this roster may still need
{need_lines}

## Confidence and limitations
Confidence is moderate for comparing profiles inside this structured feature space and low for causal transaction outcomes. The language layer did not calculate or alter the result. {limitations}
""".strip()


def generate_scouting_report(packet: dict, prefer_llm: bool = True) -> GenerationResult:
    if prefer_llm:
        try:
            response = generate_text(scouting_prompt(packet), SYSTEM_INSTRUCTIONS)
            return GenerationResult(response.text, f"openai:{response.model}")
        except LLMUnavailableError as exc:
            return GenerationResult(fallback_scouting_report(packet), "template", str(exc))
    return GenerationResult(fallback_scouting_report(packet), "template")


def generate_comparison(packet: dict, prefer_llm: bool = True) -> GenerationResult:
    if prefer_llm:
        try:
            response = generate_text(comparison_prompt(packet), SYSTEM_INSTRUCTIONS)
            return GenerationResult(response.text, f"openai:{response.model}")
        except LLMUnavailableError as exc:
            return GenerationResult(fallback_comparison(packet), "template", str(exc))
    return GenerationResult(fallback_comparison(packet), "template")


def answer_question(packet: dict, question: str, prefer_llm: bool = True) -> GenerationResult:
    if prefer_llm:
        try:
            response = generate_text(question_prompt(packet, question), SYSTEM_INSTRUCTIONS)
            return GenerationResult(response.text, f"openai:{response.model}")
        except LLMUnavailableError as exc:
            return GenerationResult(fallback_answer(packet, question), "template", str(exc))
    return GenerationResult(fallback_answer(packet, question), "template")


def generate_similarity_explanation(
    packet: dict, prefer_llm: bool = True
) -> GenerationResult:
    """Explain a fixed retrieval result with grounded generation or fallback."""

    if prefer_llm:
        try:
            response = generate_text(similarity_prompt(packet), SYSTEM_INSTRUCTIONS)
            return GenerationResult(response.text, f"openai:{response.model}")
        except LLMUnavailableError as exc:
            return GenerationResult(
                fallback_similarity_explanation(packet), "template", str(exc)
            )
    return GenerationResult(fallback_similarity_explanation(packet), "template")


def generate_team_needs_report(
    packet: dict, prefer_llm: bool = True
) -> GenerationResult:
    """Explain fixed team-needs results with grounded generation or fallback."""

    if prefer_llm:
        try:
            response = generate_text(team_needs_prompt(packet), SYSTEM_INSTRUCTIONS)
            return GenerationResult(response.text, f"openai:{response.model}")
        except LLMUnavailableError as exc:
            return GenerationResult(
                fallback_team_needs_report(packet), "template", str(exc)
            )
    return GenerationResult(fallback_team_needs_report(packet), "template")


def generate_roster_simulation_report(
    packet: dict, prefer_llm: bool = True
) -> GenerationResult:
    """Explain fixed roster-simulation results with grounded generation or fallback."""

    if prefer_llm:
        try:
            response = generate_text(roster_simulation_prompt(packet), SYSTEM_INSTRUCTIONS)
            return GenerationResult(response.text, f"openai:{response.model}")
        except LLMUnavailableError as exc:
            return GenerationResult(
                fallback_roster_simulation_report(packet), "template", str(exc)
            )
    return GenerationResult(fallback_roster_simulation_report(packet), "template")
