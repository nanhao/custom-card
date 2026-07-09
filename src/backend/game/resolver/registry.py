"""CardResolverRegistry — CardType -> list[Resolver] mapping.

Module-level registration maps every CardType to its Resolver list.
Registration runs at import time.
"""

from src.backend.game.card import CardType
from src.backend.game.resolver.attack_damage import AttackDamageResolver
from src.backend.game.resolver.base import Resolver
from src.backend.game.resolver.build import BuildResolver
from src.backend.game.resolver.flex_damage import FlexDamageResolver
from src.backend.game.resolver.gain_energy import GainEnergyResolver
from src.backend.game.resolver.grant_defense import GrantDefenseResolver
from src.backend.game.resolver.grant_guard import GrantGuardResolver
from src.backend.game.resolver.heavy_damage import HeavyDamageResolver
from src.backend.game.resolver.reflect import ReflectResolver


class CardResolverRegistry:
    """Registry mapping CardType to its Resolver list.

    All methods are classmethods. Registration happens at module load time.
    """

    _registry: dict[CardType, list[Resolver]] = {}

    @classmethod
    def register(cls, card_type: CardType, resolver: Resolver) -> None:
        """Register a resolver for the given card type (appends, not replaces)."""
        if card_type not in cls._registry:
            cls._registry[card_type] = []
        cls._registry[card_type].append(resolver)

    @classmethod
    def get_resolvers(cls, card_type: CardType) -> list[Resolver]:
        """Return all registered resolvers for card_type (unsorted).

        Caller should sort by get_priority() before invoking.
        """
        return list(cls._registry.get(card_type, []))


# ---------------------------------------------------------------------------
# Module-level registration
# ---------------------------------------------------------------------------

CardResolverRegistry.register(CardType.ATTACK, AttackDamageResolver())
CardResolverRegistry.register(CardType.DEFENSE, GrantDefenseResolver())
CardResolverRegistry.register(CardType.DEFENSE, GrantGuardResolver())
CardResolverRegistry.register(CardType.HEAVY, HeavyDamageResolver())
CardResolverRegistry.register(CardType.COUNTER, GrantDefenseResolver())
CardResolverRegistry.register(CardType.COUNTER, ReflectResolver())
CardResolverRegistry.register(CardType.ENERGY, GainEnergyResolver())
CardResolverRegistry.register(CardType.BUILD, BuildResolver())
CardResolverRegistry.register(CardType.FLEX, GrantDefenseResolver())
CardResolverRegistry.register(CardType.FLEX, FlexDamageResolver())
