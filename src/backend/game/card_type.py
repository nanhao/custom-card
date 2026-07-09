"""CardType enum — pure identifier for the custom card game v2 (design/22).

Per design/22: CardType extracted from card.py into a standalone module
to break the circular dependency between card.py and resolver/.
This module has zero dependencies on other game/ modules.

Per design/25: build_cost lookup added here to avoid circular dependency
between build_resolver.py and card_def.py.
"""

from enum import Enum, nonmember


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

    def get_build_cost(self) -> int:
        """Return the energy cost to construct this card type via BUILD.

        Per design/25: provided on CardType to avoid circular dependency —
        build_resolver.py calls this instead of card_def.get_card_def().
        Values must stay aligned with CardDef.build_cost in card_def.py.
        """
        return CardType._BUILD_COST[self.value]

    # Build cost mapping — nonmember prevents enum treatment.
    # Must stay aligned with intention/card.md and CardDef.build_cost.
    _BUILD_COST = nonmember({
        "attack": 1,
        "defense": 1,
        "heavy": 2,
        "counter": 2,
        "energy": 1,
        "build": 1,
        "flex": 2,
    })
