"""Shared app bootstrap that makes the src-layout package importable."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ball_ai.data.sample_loader import ensure_sample_database  # noqa: E402


def initialize_app() -> None:
    """Create and seed the local database on the first app run."""

    ensure_sample_database()

