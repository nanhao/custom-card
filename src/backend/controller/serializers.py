"""Serialization helpers for the REST API adapter.

Per architecture D10, this module contains NO business logic — only pure
transformation functions for CardType ↔ string, RoundResult → dict, and
error response formatting.

All card values use the `CardType.value` string (e.g. "attack", "defense").
"""

from src.backend.game.card import CardType
from src.backend.game.game_core import RoundResult

# ---------------------------------------------------------------------------
# CardType ↔ string
# ---------------------------------------------------------------------------

# Pre-built lookup for O(1) string-to-enum conversion.
_CARD_BY_VALUE: dict[str, CardType] = {c.value: c for c in CardType}


def card_to_str(card: CardType) -> str:
    """Convert a CardType enum member to its string representation.

    Returns the .value attribute (e.g. CardType.ATTACK → "attack").
    """
    return card.value


def str_to_card(value: str) -> CardType:
    """Convert a string back to a CardType enum member.

    Args:
        value: Card string value (e.g. "attack", "defense", "heavy", "counter").

    Returns:
        The corresponding CardType enum member.

    Raises:
        ValueError: If the string does not match any CardType value.
    """
    if value not in _CARD_BY_VALUE:
        raise ValueError(f"Unknown card type: {value!r}")
    return _CARD_BY_VALUE[value]


# ---------------------------------------------------------------------------
# RoundResult → dict
# ---------------------------------------------------------------------------


def round_result_to_dict(rr: RoundResult) -> dict:
    """Serialize a RoundResult to a JSON-safe dict.

    Converts all CardType lists to string lists via card_to_str().
    The resulting dict matches the round_result schema in controller_api.md §4.2.

    Args:
        rr: The RoundResult value object to serialize.

    Returns:
        dict with all RoundResult fields, with cards as string arrays.
    """
    return {
        "round": rr.round,
        "cycle": rr.cycle,
        "round_in_cycle": rr.round_in_cycle,
        "cards_1": [card_to_str(c) for c in rr.cards_1],
        "cards_2": [card_to_str(c) for c in rr.cards_2],
        "damage_1": rr.damage_1,
        "damage_2": rr.damage_2,
        "hp_1_after": rr.hp_1_after,
        "hp_2_after": rr.hp_2_after,
        "energy_1_after": rr.energy_1_after,
        "energy_2_after": rr.energy_2_after,
        "statuses_1": [ps.to_dict() for ps in rr.statuses_1],
        "statuses_2": [ps.to_dict() for ps in rr.statuses_2],
        "details": rr.details,
        "is_game_over": rr.is_game_over,
        "winner": rr.winner,
        "cycle_ended": rr.cycle_ended,
        "bleed_damage_1": rr.bleed_damage_1,
        "bleed_damage_2": rr.bleed_damage_2,
        "after_cycle": rr.after_cycle,
        "after_round_in_cycle": rr.after_round_in_cycle,
        "after_hp_1": rr.after_hp_1,
        "after_hp_2": rr.after_hp_2,
        "after_statuses_1": [ps.to_dict() for ps in rr.after_statuses_1],
        "after_statuses_2": [ps.to_dict() for ps in rr.after_statuses_2],
    }


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------


def error_response(message: str, code: str, status: int) -> dict:
    """Build a standard error response dict.

    Args:
        message: Human-readable error description.
        code: Machine-readable error code (e.g. "INVALID_CARD", "GAME_NOT_FOUND").
        status: HTTP status code (used by the route handler, not included in body).

    Returns:
        dict with keys "error" (str) and "code" (str).
    """
    return {
        "error": message,
        "code": code,
    }
