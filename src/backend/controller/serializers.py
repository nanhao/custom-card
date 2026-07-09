"""Serialization helpers for the REST API adapter.

Per architecture D10, this module contains NO business logic — only pure
transformation functions for CardType ↔ string, RoundResult → dict, and
error response formatting.

All card values use the `CardType.value` string (e.g. "attack", "defense").
"""

from src.backend.game.card_type import CardType
from src.backend.game.game_core import RoundResult
from src.backend.game.game_state import GameState

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

    Delegates to RoundResult.to_dict() which produces the nested
    before/after_resolve/end_actions/after_end_actions structure.

    Args:
        rr: The RoundResult value object to serialize.

    Returns:
        dict with nested structure matching the round_result API schema.
    """
    return RoundResult.to_dict(rr)


# ---------------------------------------------------------------------------
# GameState → dict
# ---------------------------------------------------------------------------


def game_state_to_dict(gs: GameState) -> dict:
    """Serialize a GameState to a JSON-safe dict.

    Thin wrapper that delegates to GameState.to_dict().

    Args:
        gs: The GameState value object to serialize.

    Returns:
        dict with the same structure as the old get_state() return value.
    """
    return GameState.to_dict(gs)


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
