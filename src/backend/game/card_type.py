"""CardType enum — pure identifier for the custom card game v2 (design/22).

Per design/22: CardType extracted from card.py into a standalone module
to break the circular dependency between card.py and resolver/.
This module has zero dependencies on other game/ modules.
"""

from enum import Enum


class CardType(Enum):
    """Seven card types available in the v2 game (design/11 §7).

    Values are short string identifiers used for serialization and display.
    """

    ATTACK = "attack"
    DEFENSE = "defense"
    HEAVY = "heavy"
    COUNTER = "counter"
    ENERGY = "energy"
    BUILD = "build"
    FLEX = "flex"
