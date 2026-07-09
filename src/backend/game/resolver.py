"""Priority-layered card resolution engine with strategy pattern (design/21).

Per design/21: _process_effect module-level function is replaced by per-effect
Resolver subclasses registered via CardResolverRegistry. Each Resolver directly
modifies GameCore player state through its public methods.

Resolvers receive all context via ResolveContext dataclass — no indirect
GameCore state queries needed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.backend.game.card import CardType
from src.backend.game.player import Player, PlayerHistoryItem
from src.backend.game.player_state import StatusType


# =============================================================================
# Data types
# =============================================================================


@dataclass
class ResolveDetails:
    """Settlement detail collector, the data source for ResolutionResult.details."""

    details: list[str] = field(default_factory=list)

    def add(self, detail: str) -> None:
        """Record a resolution step. Empty strings are ignored."""
        if detail:
            self.details.append(detail)


@dataclass
class ResolutionResult:
    """Resolution outcome. State changes are already applied to GameCore players,
    so this only carries the details from ResolveDetails.
    """

    details: list[str]


@dataclass
class ResolveContext:
    """All context needed by a Resolver to settle one card's effects.

    Passed directly to resolve() — no GameCore indirection needed.
    """

    card_owner: Player
    opponent: Player
    card_played_by_owner: CardType
    card_played_by_opponent: CardType


# =============================================================================
# Resolver — abstract base class
# =============================================================================


class Resolver(ABC):
    """Abstract base for card effect resolvers.

    Each subclass encapsulates a single effect resolution logic. Subclasses
    are stateless singletons registered via CardResolverRegistry.
    """

    @abstractmethod
    def get_priority(self) -> int:
        """Return resolution priority."""

    @abstractmethod
    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Execute effect resolution, directly modifying Player state via ctx.

        Records settlement details via details.add().
        """


# =============================================================================
# Concrete Resolver subclasses — one per card effect
# =============================================================================


class AttackDamageResolver(Resolver):
    """Settles ATTACK damage (priority 10).

    Damage = min(streak + 1, 3), where streak is computed from
    card_owner's history via CardResolver.get_consecutive_attack().
    Blocked by opponent DEFENSE status.
    """

    # Maximum damage from consecutive attack streak
    MAX_STREAK_DAMAGE = 3

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        # Compute streak from card_owner's history at resolution time
        streak = CardResolver.get_consecutive_attack(ctx.card_owner.history)
        raw_damage = min(streak + 1, self.MAX_STREAK_DAMAGE)

        if ctx.opponent.has_status(StatusType.DEFENSE):
            details.add(f"Attack(streak={streak}) blocked by opponent defense")
        else:
            ctx.opponent.take_damage(raw_damage)
            details.add(f"Attack(streak={streak}) deals {raw_damage} to opponent")


class GrantDefenseResolver(Resolver):
    """Grants DEFENSE status to the card_owner (priority 11)."""

    def get_priority(self) -> int:
        return 11

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        ctx.card_owner.grant_status(StatusType.DEFENSE)
        details.add("gains DEFENSE")


class GrantGuardResolver(Resolver):
    """Grants GUARD status if card_owner had DEFENSE pre-round (priority 12).

    Priority 12 runs BEFORE any GrantDefenseResolver (priority 11), so
    card_owner.has_status(DEFENSE) reflects the pre-round state.
    Does NOT grant GUARD if already present.
    """

    def get_priority(self) -> int:
        return 12

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        # Priority 12 ensures no GrantDefense has run yet,
        # so DEFENSE status reflects pre-round state.
        if not ctx.card_owner.has_status(StatusType.DEFENSE):
            return

        # Avoid duplicate GUARD
        if ctx.card_owner.has_status(StatusType.GUARD):
            return

        ctx.card_owner.grant_status(StatusType.GUARD)
        details.add("gains GUARD (consecutive defense)")


class HeavyDamageResolver(Resolver):
    """Settles HEAVY strike damage (priority 10).

    Deals 1 damage, blocked by opponent GUARD status only.
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        if ctx.opponent.has_status(StatusType.GUARD):
            details.add("Heavy Strike blocked by opponent GUARD")
        else:
            ctx.opponent.take_damage(1)
            details.add("Heavy Strike deals 1 to opponent")


class ReflectResolver(Resolver):
    """Settles COUNTER reflect damage (priority 10).

    Reflects 1 damage if opponent played ATTACK or HEAVY.
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        if ctx.card_played_by_opponent not in (CardType.ATTACK, CardType.HEAVY):
            return

        ctx.opponent.take_damage(1)
        details.add(
            f"Counter reflects {ctx.card_played_by_opponent.value}, deals 1 to opponent"
        )


class GainEnergyResolver(Resolver):
    """Grants 1 energy to the card_owner (priority 10).

    Clamping to [0,5] is handled by Player.add_energy().
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        ctx.card_owner.add_energy(1)
        details.add("gains 1 energy")


class FlexDamageResolver(Resolver):
    """Settles FLEX damage portion (priority 10).

    Deals 1 damage if opponent does NOT have DEFENSE status.
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        if ctx.opponent.has_status(StatusType.DEFENSE):
            details.add("FLEX damage blocked by opponent defense")
        else:
            ctx.opponent.take_damage(1)
            details.add("FLEX damage deals 1 to opponent")


class BuildResolver(Resolver):
    """No immediate effect — BUILD construction handled by GameCore (priority 10)."""

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        details.add("BUILD — no immediate effect")


# =============================================================================
# CardResolverRegistry — CardType -> list[Resolver]
# =============================================================================


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
# Module-level registration — maps every CardType to its Resolver list
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


# =============================================================================
# CardResolver — priority-layered resolution orchestrator
# =============================================================================


class CardResolver:
    """Priority-layered resolution orchestrator.

    Queries CardResolverRegistry for each card's resolvers, builds a
    ResolveContext per card, sorts by priority, and delegates to each Resolver.
    """

    @staticmethod
    def get_consecutive_attack(history: list[PlayerHistoryItem]) -> int:
        """Count consecutive ATTACK cards from the end of history."""
        count = 0
        for item in reversed(history):
            if item.card == CardType.ATTACK:
                count += 1
            else:
                break
        return count

    @staticmethod
    def resolve(
        game_core,  # GameCore instance (no type annotation — avoids circular import)
        cards_1: list[CardType],
        cards_2: list[CardType],
    ) -> ResolutionResult:
        """Orchestrate resolution of both players' cards.

        Flow:
        1. For each card, build a ResolveContext with owner/opponent/cards.
        2. Query CardResolverRegistry for resolvers.
        3. Flatten into (resolver, ctx) pairs.
        4. Sort by resolver.get_priority() (highest first).
        5. Invoke each resolver.resolve(ctx, details).
        """
        details = ResolveDetails()

        # Build (resolver, ResolveContext) tasks for both players
        tasks: list[tuple[Resolver, ResolveContext]] = []

        for card_type in cards_1:
            ctx = ResolveContext(
                card_owner=game_core.player_1,
                opponent=game_core.player_2,
                card_played_by_owner=card_type,
                card_played_by_opponent=cards_2[0],
            )
            for resolver in CardResolverRegistry.get_resolvers(card_type):
                tasks.append((resolver, ctx))

        for card_type in cards_2:
            ctx = ResolveContext(
                card_owner=game_core.player_2,
                opponent=game_core.player_1,
                card_played_by_owner=card_type,
                card_played_by_opponent=cards_1[0],
            )
            for resolver in CardResolverRegistry.get_resolvers(card_type):
                tasks.append((resolver, ctx))

        # Sort by priority: highest first
        tasks.sort(key=lambda t: t[0].get_priority(), reverse=True)

        for resolver, ctx in tasks:
            resolver.resolve(ctx, details)

        return ResolutionResult(details=list(details.details))
