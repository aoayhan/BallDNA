"""Pure filtering helpers for player-selection interfaces."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

import pandas as pd


def _normalized_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value))
    return "".join(character for character in decomposed if character.isalnum()).lower()


def _edit_distance(left: str, right: str) -> int:
    """Return Levenshtein distance without adding a fuzzy-search dependency."""

    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row_index, right_character in enumerate(right, start=1):
        current = [row_index]
        for column_index, left_character in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1]
                    + int(left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def strict_player_name_matches(
    players: pd.DataFrame,
    query: str,
    *,
    maximum_distance: int = 2,
) -> pd.DataFrame:
    """Find literal or near-spelling player names without broad fuzzy matches.

    Literal name fragments rank first. Otherwise the query may differ from a
    full name or individual name token by at most one character for very short
    queries and ``maximum_distance`` for queries of four or more characters.
    """

    if "player_name" not in players:
        raise ValueError("Player data is missing column: player_name")
    normalized_query = _normalized_name(query)
    if not normalized_query:
        return players.copy().reset_index(drop=True)
    allowed_distance = 1 if len(normalized_query) <= 3 else maximum_distance
    rows = []
    for row in players.itertuples(index=False):
        raw_name = str(row.player_name)
        compact_name = _normalized_name(raw_name)
        tokens = [_normalized_name(token) for token in raw_name.split()]
        if normalized_query == compact_name or normalized_query in tokens:
            priority = 0
            distance = 0
        elif any(token.startswith(normalized_query) for token in tokens):
            priority = 1
            distance = 0
        elif normalized_query in compact_name:
            priority = 2
            distance = 0
        else:
            priority = 3
            distance = min(
                _edit_distance(normalized_query, candidate)
                for candidate in [compact_name, *tokens]
                if candidate
            )
        if distance <= allowed_distance:
            item = row._asdict()
            item["name_match_priority"] = priority
            item["name_match_distance"] = distance
            rows.append(item)
    if not rows:
        return players.head(0).assign(
            name_match_priority=pd.Series(dtype=int),
            name_match_distance=pd.Series(dtype=int),
        )
    result = pd.DataFrame(rows)
    if result["name_match_priority"].lt(3).any():
        result = result.loc[result["name_match_priority"].lt(3)]
    return result.sort_values(
        ["name_match_priority", "name_match_distance", "player_name"], kind="stable"
    ).reset_index(drop=True)


@dataclass(frozen=True)
class PlayerSearchResult:
    """Player matches plus whether the team filter had to be relaxed."""

    matches: pd.DataFrame
    used_league_wide_fallback: bool = False


def filter_players(
    players: pd.DataFrame,
    team: str | None = None,
    query: str = "",
) -> pd.DataFrame:
    """Filter players by an optional team and case-insensitive name fragment."""

    required = {"player_id", "player_name", "team", "position"}
    if missing := required - set(players.columns):
        raise ValueError(f"Player data is missing columns: {sorted(missing)}")

    filtered = players.copy()
    if team:
        filtered = filtered.loc[filtered["team"].eq(team)]

    normalized_query = query.strip()
    if normalized_query:
        filtered = filtered.loc[
            filtered["player_name"].str.contains(
                normalized_query, case=False, regex=False, na=False
            )
        ]

    return filtered.reset_index(drop=True)


def resolve_player_search(
    players: pd.DataFrame,
    team: str | None = None,
    query: str = "",
) -> PlayerSearchResult:
    """Search the selected team first, then recommend league-wide name matches."""

    matches = strict_player_name_matches(filter_players(players, team=team), query)
    if team and query.strip():
        alternatives = strict_player_name_matches(players, query)
        local_score = (
            tuple(matches.iloc[0][["name_match_priority", "name_match_distance"]])
            if not matches.empty else (float("inf"), float("inf"))
        )
        league_score = (
            tuple(alternatives.iloc[0][["name_match_priority", "name_match_distance"]])
            if not alternatives.empty else (float("inf"), float("inf"))
        )
        if league_score < local_score:
            return PlayerSearchResult(alternatives, used_league_wide_fallback=True)
    return PlayerSearchResult(matches)


def order_player_candidates(
    players: pd.DataFrame,
    *,
    preferred_team: str | None = None,
    excluded_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Prepare a typeahead list, strictly restricted to a selected team."""

    required = {"player_id", "player_name", "team"}
    if missing := required - set(players.columns):
        raise ValueError(f"Player data is missing columns: {sorted(missing)}")
    excluded = set(int(player_id) for player_id in (excluded_ids or []))
    candidates = players.loc[
        ~players["player_id"].astype(int).isin(excluded)
    ].copy()
    if preferred_team:
        candidates = candidates.loc[candidates["team"].eq(preferred_team)].copy()
    candidates["team_priority"] = 0
    return candidates.sort_values(
        ["team_priority", "player_name", "team"], kind="stable"
    ).reset_index(drop=True)
