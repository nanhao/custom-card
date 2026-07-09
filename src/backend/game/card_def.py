"""CardDef dataclass and static card metadata table (design/22).

Per design/22: CardDef gains a `resolvers` field that stores Resolver
subclass references. CardResolverRegistry has been deleted — the mapping
is now defined inline in _CARD_DEFS.

CardType was extracted to card_type.py to break the circular dependency
with resolver/.
"""

from dataclasses import dataclass

from src.backend.game.card_type import CardType
from src.backend.game.resolver.attack_damage_resolver import AttackDamageResolver
from src.backend.game.resolver.build_resolver import BuildResolver
from src.backend.game.resolver.flex_damage_resolver import FlexDamageResolver
from src.backend.game.resolver.gain_energy_resolver import GainEnergyResolver
from src.backend.game.resolver.grant_defense_resolver import GrantDefenseResolver
from src.backend.game.resolver.grant_guard_resolver import GrantGuardResolver
from src.backend.game.resolver.heavy_damage_resolver import HeavyDamageResolver
from src.backend.game.resolver.reflect_resolver import ReflectResolver


@dataclass(frozen=True)
class CardDef:
    """Per-card static metadata (design/22, design/23).

    Per design/21: effects and priority fields removed — effect logic
    encapsulated in Resolver subclasses.
    Per design/22: resolvers field added — Resolver subclass references
    replace the deleted CardResolverRegistry.
    Per design/23: icon field removed — display logic moved to frontend.

    Attributes:
        card_type: The CardType enum member this definition belongs to.
        build_cost: Energy cost (1 or 2) to construct this card via BUILD.
        display_name: Human-readable Chinese name for UI display.
        resolvers: Resolver classes for this card, in registration order.
    """

    card_type: CardType
    build_cost: int
    display_name: str
    resolvers: tuple[type, ...]


# ---------------------------------------------------------------------------
# Static card metadata table — one CardDef per CardType.
# resolvers field replaces CardResolverRegistry (design/22).
# ---------------------------------------------------------------------------

_CARD_DEFS: dict[CardType, CardDef] = {
    CardType.ATTACK: CardDef(
        card_type=CardType.ATTACK,
        build_cost=1,
        display_name="攻击",
        resolvers=(AttackDamageResolver,),
    ),
    CardType.DEFENSE: CardDef(
        card_type=CardType.DEFENSE,
        build_cost=1,
        display_name="防御",
        resolvers=(GrantDefenseResolver, GrantGuardResolver),
    ),
    CardType.HEAVY: CardDef(
        card_type=CardType.HEAVY,
        build_cost=1,
        display_name="重击",
        resolvers=(HeavyDamageResolver,),
    ),
    CardType.COUNTER: CardDef(
        card_type=CardType.COUNTER,
        build_cost=2,
        display_name="反击",
        resolvers=(GrantDefenseResolver, ReflectResolver),
    ),
    CardType.ENERGY: CardDef(
        card_type=CardType.ENERGY,
        build_cost=1,
        display_name="聚能",
        resolvers=(GainEnergyResolver,),
    ),
    CardType.BUILD: CardDef(
        card_type=CardType.BUILD,
        build_cost=1,
        display_name="构建",
        resolvers=(BuildResolver,),
    ),
    CardType.FLEX: CardDef(
        card_type=CardType.FLEX,
        build_cost=2,
        display_name="攻守兼备",
        resolvers=(GrantDefenseResolver, FlexDamageResolver),
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_card_def(card_type: CardType) -> CardDef:
    """Return the CardDef for a given CardType.

    Per design/22: the returned CardDef includes resolver class references
    in the `resolvers` field, replacing the deleted CardResolverRegistry.

    Args:
        card_type: Any of the 7 CardType enum members.

    Returns:
        Immutable CardDef with card_type, build_cost, display_name,
        resolvers.
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
