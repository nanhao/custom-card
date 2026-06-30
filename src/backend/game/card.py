"""Card type definitions for the custom card game.

This module defines the CardType enum representing the four card types
and a factory function to produce the standard 7-card starting hand.
Per architecture D9, this module is pure logic — no UI, no I/O.
"""

from enum import Enum


class CardType(Enum):
    """The four card types available in the game.

    Values are short string identifiers used for serialization and display.
    """

    ATTACK = "attack"
    DEFENSE = "defense"
    HEAVY_STRIKE = "heavy"
    COUNTER = "counter"


def get_initial_hand() -> list[CardType]:
    """Return a new copy of the standard 7-card starting hand.

    Composition: 3×ATTACK, 2×DEFENSE, 1×HEAVY_STRIKE, 1×COUNTER.
    Returns a fresh list on each call to prevent shared-mutation bugs
    (architecture D2: Player hand is independently mutable).
    """
    return [
        CardType.ATTACK,
        CardType.ATTACK,
        CardType.ATTACK,
        CardType.DEFENSE,
        CardType.DEFENSE,
        CardType.HEAVY_STRIKE,
        CardType.COUNTER,
    ]
