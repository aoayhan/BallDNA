"""Regression checks for clearable player selectors in the redesigned app."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize(
    ("view", "state_key", "message"),
    [
        ("compare", "mock_compare_a", "Choose Player A to continue."),
        ("compare", "mock_compare_b", "Choose Player B to continue."),
        ("discover", "mock_player", "Choose a player to continue."),
    ],
)
def test_cleared_player_selection_is_recoverable(
    view: str, state_key: str, message: str
) -> None:
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app/main.py"), default_timeout=60
    )
    app.query_params["view"] = view
    app.session_state[state_key] = None
    app.run()

    assert not app.exception
    assert message in [element.value for element in app.info]


def test_player_selector_labels_are_unique() -> None:
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app/main.py"), default_timeout=60
    )
    app.query_params["view"] = "compare"
    app.run()

    options = app.selectbox[0].options
    assert len(options) == len(set(options))
    assert sum(option.startswith("Mike James · ") for option in options) == 2


def test_compare_rejects_stale_same_player_state() -> None:
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app/main.py"), default_timeout=60
    )
    app.query_params["view"] = "compare"
    app.session_state["mock_compare_a"] = "Allen Iverson"
    app.session_state["mock_compare_b"] = "Allen Iverson"
    app.run()

    assert not app.exception
    assert "Choose Player B to continue." in [element.value for element in app.info]
