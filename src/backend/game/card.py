"""Card type definitions for the custom card game v2 (design/11, design/21).

This module defines the CardType enum (7 values) and the CardDef dataclass
holding per-card static metadata (build_cost, display_name, icon).

Per design/21: EffectType enum and CardDef.effects/priority fields removed.
Effect logic is encapsulated in resolver.py Resolver subclasses.

Per architecture D1, CardType serves as identifier only; CardDef carries
card metadata. Effect specifications moved to CardResolverRegistry.
"""

from dataclasses import dataclass
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


@dataclass(frozen=True)
class CardDef:
    """Per-card static metadata (design/21).

    Per design/21: effects and priority fields removed — effect logic
    encapsulated in Resolver subclasses via CardResolverRegistry.

    Attributes:
        card_type: The CardType enum member this definition belongs to.
        build_cost: Energy cost (1 or 2) to construct this card via BUILD.
        display_name: Human-readable Chinese name for UI display.
        icon: Emoji icon for UI display.
    """

    card_type: CardType
    build_cost: int
    display_name: str
    icon: str


# ---------------------------------------------------------------------------
# Static card metadata table — one CardDef per CardType.
# ---------------------------------------------------------------------------

_CARD_DEFS: dict[CardType, CardDef] = {
    CardType.ATTACK: CardDef(
        card_type=CardType.ATTACK,
        build_cost=1,
        display_name="攻击",
        icon="⚔️",
    ),
    CardType.DEFENSE: CardDef(
        card_type=CardType.DEFENSE,
        build_cost=1,
        display_name="防御",
        icon="🛡️",
    ),
    CardType.HEAVY: CardDef(
        card_type=CardType.HEAVY,
        build_cost=1,
        display_name="重击",
        icon="💥",
    ),
    CardType.COUNTER: CardDef(
        card_type=CardType.COUNTER,
        build_cost=2,
        display_name="反击",
        icon="↩️",
    ),
    CardType.ENERGY: CardDef(
        card_type=CardType.ENERGY,
        build_cost=1,
        display_name="聚能",
        icon="⚡",
    ),
    CardType.BUILD: CardDef(
        card_type=CardType.BUILD,
        build_cost=1,
        display_name="构建",
        icon="🔨",
    ),
    CardType.FLEX: CardDef(
        card_type=CardType.FLEX,
        build_cost=2,
        display_name="攻守兼备",
        icon="⚖️",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_card_def(card_type: CardType) -> CardDef:
    """Return the CardDef for a given CardType.

    Args:
        card_type: Any of the 7 CardType enum members.

    Returns:
        Immutable CardDef with priority, effects, build_cost, etc.
    """
    return _CARD_DEFS[card_type]


def get_initial_draw_pile() -> list[CardType]:
    """Return the standard 7-card initial draw pile (fresh copy).

    Composition: 2×ATTACK, 1 each of DEFENSE/HEAVY/COUNTER/ENERGY/BUILD.
    FLEX is not included in the initial draw pile.

    Returns a fresh list on each call to prevent shared-mutation bugs.
    """
    return [
        CardType.ATTACK,
        CardType.ATTACK,
        CardType.DEFENSE,
        CardType.HEAVY,
        CardType.COUNTER,
        CardType.ENERGY,
        CardType.BUILD,
    ]
