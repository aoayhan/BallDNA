"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from ball_ai.config import settings
from ball_ai.data.sample_loader import load_sample_data


@pytest.fixture()
def demo_database(tmp_path: Path) -> Path:
    path = tmp_path / "test_hooplens.db"
    load_sample_data(path, sample_dir=settings.sample_data_dir, force=True)
    return path

