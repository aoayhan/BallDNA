"""Run lightweight product-quality checks against deterministic reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ball_ai.ai.grounding import build_player_evidence  # noqa: E402
from ball_ai.ai.quality import evaluate_report  # noqa: E402
from ball_ai.ai.report_generator import generate_scouting_report  # noqa: E402
from ball_ai.data.database import (  # noqa: E402
    get_data_status_text,
    get_player_profile,
    get_players,
    get_recent_games,
)
from ball_ai.data.sample_loader import ensure_sample_database  # noqa: E402


def main() -> int:
    ensure_sample_database()
    results = {}
    for row in get_players().head(5).itertuples():
        profile = get_player_profile(int(row.player_id))
        packet = build_player_evidence(
            profile,
            get_recent_games(int(row.player_id)),
            get_data_status_text(),
        )
        report = generate_scouting_report(packet, prefer_llm=False).text
        results[row.player_name] = evaluate_report(report, packet)

    print(json.dumps(results, indent=2))
    passed = all(
        check["passed"]
        for player_checks in results.values()
        for check in player_checks.values()
    )
    print(f"\nOverall: {'PASS' if passed else 'REVIEW'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
