"""Resolver subpackage — card effect resolution engine (design/22).

Per design/22: CardResolverRegistry removed; card ⇄ resolver mapping
now lives in CardDef.resolvers (card_def.py).

Re-exports all public symbols so that external code can continue to use:
    from src.backend.game.resolver import CardResolver
"""

from src.backend.game.resolver.base import (
    ResolveContext,
    ResolveDetails,
    ResolutionResult,
    Resolver,
)
from src.backend.game.resolver.orchestrator import CardResolver

# Individual resolver classes — for direct testing
from src.backend.game.resolver.attack_damage_resolver import AttackDamageResolver
from src.backend.game.resolver.build_resolver import BuildResolver
from src.backend.game.resolver.flex_damage_resolver import FlexDamageResolver
from src.backend.game.resolver.gain_energy_resolver import GainEnergyResolver
from src.backend.game.resolver.grant_defense_resolver import GrantDefenseResolver
from src.backend.game.resolver.grant_guard_resolver import GrantGuardResolver
from src.backend.game.resolver.grant_fragile_resolver import GrantFragileResolver
from src.backend.game.resolver.heavy_damage_resolver import HeavyDamageResolver
from src.backend.game.resolver.reflect_resolver import ReflectResolver
