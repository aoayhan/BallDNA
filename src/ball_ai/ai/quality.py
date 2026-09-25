"""Lightweight product-quality checks for deterministic scouting summaries."""

from __future__ import annotations

import json
import re

NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?")
REPORT_SECTIONS = [
    "Summary",
    "Strengths",
    "Weaknesses or risks",
    "Recent trend",
    "Best role fit",
    "Evidence used",
    "Confidence and limitations",
]


def required_sections_present(report_text: str) -> tuple[bool, list[str]]:
    missing = [section for section in REPORT_SECTIONS if f"## {section}" not in report_text]
    return not missing, missing


def contains_evidence_citations(report_text: str) -> bool:
    return bool(re.search(r"\[E\d+\]", report_text))


def unsupported_numbers(report_text: str, packet: dict) -> list[str]:
    """Flag numeric tokens that do not appear anywhere in the supplied packet.

    This is a heuristic guardrail, not a semantic factuality evaluator.
    """

    packet_text = json.dumps(packet)
    allowed = {token.rstrip("%") for token in NUMBER_PATTERN.findall(packet_text)}

    # Reports round evidence values for readability, and store percentages as
    # decimals while displaying them as human-readable percentages. Permit those
    # deterministic renderings without allowing arbitrary numbers.
    def add_report_renderings(value) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                add_report_renderings(nested)
        elif isinstance(value, list):
            for nested in value:
                add_report_renderings(nested)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            allowed.update({f"{numeric:.1f}", f"{numeric:.2f}"})
            if -1 < numeric < 1 and numeric != 0:
                percentage = numeric * 100
                allowed.update(
                    {
                        f"{percentage:.1f}",
                        f"{percentage:.2f}",
                        f"{abs(percentage):.1f}",
                        f"{abs(percentage):.2f}",
                    }
                )

    add_report_renderings(packet)
    found = {token.rstrip("%") for token in NUMBER_PATTERN.findall(report_text)}
    return sorted(found - allowed)


def evaluate_report(report_text: str, packet: dict) -> dict:
    """Return recruiter-readable pass/fail checks plus diagnostic details."""

    sections_ok, missing = required_sections_present(report_text)
    unsupported = unsupported_numbers(report_text, packet)
    return {
        "required_sections": {"passed": sections_ok, "details": missing},
        "evidence_citations": {
            "passed": contains_evidence_citations(report_text),
            "details": [],
        },
        "numeric_grounding": {"passed": not unsupported, "details": unsupported},
        "non_empty": {"passed": bool(report_text.strip()), "details": []},
    }
