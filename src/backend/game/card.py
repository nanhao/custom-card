"""Card type definitions for the custom card game v2 (design/11).

This module defines the CardType enum (7 values), the EffectType enum (7 values),
and the CardDef dataclass holding per-card static metadata (priority, effects,
build_cost, etc.).

Per architecture D1, CardType serves as identifier only; CardDef carries
the full effect specification used by CardResolver.
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


class EffectType(Enum):
    """All card effect types used by CardResolver.

    Values are short string identifiers used for serialization.
    Per report 2026-07-08: eliminates bare-string effect_type comparisons
    in resolver.py by providing compile-time-checkable enum members.
    """

    DAMAGE = "damage"
    GRANT_DEFENSE = "grant_defense"
    GRANT_GUARD = "grant_guard"
    REFLECT = "reflect"
    GAIN_ENERGY = "gain_energy"
    FLEX_DAMAGE = "flex_damage"
    BUILD = "build"


@dataclass(frozen=True)
class CardDef:
    """Per-card static metadata (design/11 interfaces.md §1).

    Each card type has a fixed CardDef instance describing its effects,
    priority, build cost, display name, and icon.

    Attributes:
        card_type: The CardType enum member this definition belongs to.
        priority: Primary priority (10 or 11). None for multi-priority cards (FLEX).
        effects: Tuple of (effect_type, priority) pairs. Each pair defines one
                 effect unit in the priority-layered resolution engine.
        build_cost: Energy cost (1 or 2) to construct this card via BUILD.
        display_name: Human-readable Chinese name for UI display.
        icon: Emoji icon for UI display.
    """

    card_type: CardType
    priority: int | None
    effects: tuple[tuple[EffectType, int], ...]
    build_cost: int
    display_name: str
    icon: str


# ---------------------------------------------------------------------------
# Static card metadata table — one CardDef per CardType.
# ---------------------------------------------------------------------------

_CARD_DEFS: dict[CardType, CardDef] = {
    CardType.ATTACK: CardDef(
        card_type=CardType.ATTACK,
        priority=10,
        effects=((EffectType.DAMAGE, 10),),
        build_cost=1,
        display_name="攻击",
        icon="⚔️",
    ),
    CardType.DEFENSE: CardDef(
        card_type=CardType.DEFENSE,
        priority=11,
        effects=((EffectType.GRANT_DEFENSE, 11), (EffectType.GRANT_GUARD, 11)),
        build_cost=1,
        display_name="防御",
        icon="🛡️",
    ),
    CardType.HEAVY: CardDef(
        card_type=CardType.HEAVY,
        priority=10,
        effects=((EffectType.DAMAGE, 10),),
        build_cost=1,
        display_name="重击",
        icon="💥",
    ),
    CardType.COUNTER: CardDef(
        card_type=CardType.COUNTER,
        priority=11,
        effects=((EffectType.GRANT_DEFENSE, 11), (EffectType.REFLECT, 10)),
        build_cost=2,
        display_name="反击",
        icon="↩️",
    ),
    CardType.ENERGY: CardDef(
        card_type=CardType.ENERGY,
        priority=10,
        effects=((EffectType.GAIN_ENERGY, 10),),
        build_cost=1,
        display_name="聚能",
        icon="⚡",
    ),
    CardType.BUILD: CardDef(
        card_type=CardType.BUILD,
        priority=10,
        effects=((EffectType.BUILD, 10),),
        build_cost=1,
        display_name="构建",
        icon="🔨",
    ),
    CardType.FLEX: CardDef(
        card_type=CardType.FLEX,
        priority=None,  # Multi-priority: effects carry their own priorities
        effects=((EffectType.GRANT_DEFENSE, 11), (EffectType.FLEX_DAMAGE, 10)),
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
