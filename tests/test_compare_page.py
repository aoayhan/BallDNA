"""Regression check for historical players on the comparison page."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_compare_page_supports_retired_players() -> None:
    page = Path(__file__).parents[1] / "app/pages/2_Compare_Players.py"
    app = AppTest.from_file(str(page), default_timeout=30).run()
    app.selectbox[0].set_value(200755)  # JJ Redick
    app.selectbox[1].set_value(397)  # Reggie Miller
    app.run()
    app.selectbox[4].set_value("2009-10")
    app.selectbox[5].set_value("2002-03")
    app.run()

    assert not app.exception
    assert app.selectbox[0].value == 200755
    assert app.selectbox[1].value == 397
    lower_tables = "".join(element.proto.body for element in app.get("html")[3:])
    assert "2009-10" in lower_tables and "2002-03" in lower_tables
    assert "2020-21" not in lower_tables and "2004-05" not in lower_tables
