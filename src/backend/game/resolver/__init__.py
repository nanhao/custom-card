"""Resolver subpackage — card effect resolution engine (design/21).

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
from src.backend.game.resolver.registry import CardResolverRegistry

# Individual resolver classes — for direct testing
from src.backend.game.resolver.attack_damage import AttackDamageResolver
from src.backend.game.resolver.build import BuildResolver
from src.backend.game.resolver.flex_damage import FlexDamageResolver
from src.backend.game.resolver.gain_energy import GainEnergyResolver
from src.backend.game.resolver.grant_defense import GrantDefenseResolver
from src.backend.game.resolver.grant_guard import GrantGuardResolver
from src.backend.game.resolver.heavy_damage import HeavyDamageResolver
from src.backend.game.resolver.reflect import ReflectResolver
