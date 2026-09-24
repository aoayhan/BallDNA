"""Versioned prompt templates for evidence-constrained generation."""

from __future__ import annotations

import json


SYSTEM_INSTRUCTIONS = """You are BallDNA, an evidence-first basketball analytics copilot.
Use only the JSON evidence packet supplied by the application. Do not use prior
knowledge about a player, team, game, injury, or league average. Every basketball
claim must cite one or more evidence IDs in square brackets, for example [E1].
Do not invent statistics. State uncertainty when the sample is small or context is
missing. Keep the result concise and practical. The data packet may identify demo
or synthetic data; preserve that limitation and never present it as live data."""


REPORT_SECTIONS = [
    "Summary",
    "Strengths",
    "Weaknesses or risks",
    "Recent trend",
    "Best role fit",
    "Evidence used",
    "Confidence and limitations",
]


def scouting_prompt(packet: dict) -> str:
    sections = "\n".join(f"## {section}" for section in REPORT_SECTIONS)
    return f"""Write a scouting report with exactly these Markdown sections:
{sections}

Under Evidence used, list the evidence IDs cited and what each supports. Make role
fit conditional on the statistical profile; do not imply film-based knowledge.

EVIDENCE_PACKET_JSON:
{json.dumps(packet, indent=2)}"""


def comparison_prompt(packet: dict) -> str:
    return f"""Compare the two players using only the supplied packets. Use Markdown
sections: Summary, Similarities, Differences, Need-based fit, Evidence used, and
Confidence and limitations. Prefix citations with the player name because E IDs
restart in each packet. Do not declare an overall better player without a stated need.

COMPARISON_PACKET_JSON:
{json.dumps(packet, indent=2)}"""


def question_prompt(packet: dict, question: str) -> str:
    return f"""Answer the user's question in at most four short paragraphs. Start
with a direct answer, cite evidence IDs for every claim, and finish with a one-line
limitation. If the evidence cannot answer the question, say exactly what is missing.

QUESTION: {question}

EVIDENCE_PACKET_JSON:
{json.dumps(packet, indent=2)}"""


def similarity_prompt(packet: dict) -> str:
    return f"""Explain the retrieved player matches using exactly these Markdown
sections: Retrieval result, Why the matches are similar, Important differences,
and Confidence and limitations. Explain the best three matches. Cite the supplied
M#E# evidence IDs for every statistical claim. Do not recalculate or change the
ranking. Make clear that deterministic vector retrieval selected the matches and
the language model only translates the supplied distances into readable analysis.

SIMILARITY_PACKET_JSON:
{json.dumps(packet, indent=2)}"""


def team_needs_prompt(packet: dict) -> str:
    return f"""Explain the supplied deterministic team-needs analysis using exactly
these Markdown sections: Executive read, What elite teams share, Priority gaps,
Needed player archetype, Candidate examples, and Confidence and limitations.
Cite C#, G#, P#, or R# evidence IDs for every analytical claim. Do not recalculate,
reorder, add candidates, predict wins, or imply that a player is available. Clearly
separate statistical fit from causal roster advice.

TEAM_NEEDS_PACKET_JSON:
{json.dumps(packet, indent=2)}"""


def roster_simulation_prompt(packet: dict) -> str:
    return f"""Explain the supplied fixed roster simulation using exactly these
Markdown sections: Simulation read, Identity changes, Historical neighbours,
What this roster may still need, and Confidence and limitations. Cite Q#, C#,
P#, N#, R#, S#, T#, U#, V#, or W# evidence IDs for every analytical claim. Do not recalculate the model,
invent players, imply transaction feasibility, or present the win range as a
forecast guarantee. Clearly state that this is a same-season roster-profile
counterfactual and that the language model only explains fixed numeric results.
If change_detection says no detectable change, do not describe either roster as
better or worse even when the raw estimate has a small positive or negative sign.
If model_support says the roster is unsupported, never infer, reconstruct, or
mention a Net Rating or win estimate. Explain that the identity embedding remains
descriptive, cite [S1], and distinguish the talent percentile [T1] and experimental
expected-wins output [U1] from a historically calibrated forecast. Describe U1 as
the monotonic model's point estimate for roster comparison, never as validated or
likely.
If counterfactual_validation says outcome_validated is false, do not claim that
the move improves or worsens expected Net Rating or wins. State that transaction
effects are not yet backtested [V1], then explain identity, automatic role allocation,
and upside only. When an added player receives few automatic minutes, explain that
the player did not displace a lower-impact incumbent at an overlapping position;
cite the matching R# row and do not call the player intrinsically poor.

ROSTER_SIMULATION_PACKET_JSON:
{json.dumps(packet, indent=2)}"""
