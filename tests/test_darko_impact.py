"""Checks for the full-history DARKO snapshot normalizer."""

from scripts.fetch_darko_impact import _normalize_history


def test_normalize_darko_history_maps_ending_year_and_impact_components() -> None:
    result = _normalize_history([{
        "nba_id": "42", "player_name": "Example Player", "season": 1997,
        "team_name": "Example Team", "dpm": 1.0, "o_dpm": 1.5, "d_dpm": -0.5,
        "box_odpm": 1.2, "box_ddpm": -0.2,
        "on_off_odpm": 0.8, "on_off_ddpm": 0.1,
    }]).iloc[0]

    assert result["season"] == "1996-97"
    assert result["player_id"] == 42
    assert result["box_dpm"] == 1.0
    assert result["on_off_dpm"] == 0.9
