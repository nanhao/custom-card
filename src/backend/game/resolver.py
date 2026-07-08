"""Priority-layered card resolution engine with strategy pattern (design/21).

Per design/21: _process_effect module-level function is replaced by per-effect
Resolver subclasses registered via CardResolverRegistry. Each Resolver directly
modifies GameCore player state through its public methods.

Per architecture D9, this module is pure logic — no UI, no I/O.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.backend.game.card import CardType
from src.backend.game.player import PlayerHistoryItem
from src.backend.game.player_state import StatusType, PlayerStatus


@dataclass
class ResolveDetails:
    """Settlement detail collector, the data source for ResolutionResult.details.

    Each Resolver records resolution steps via add() during settlement.
    """

    details: list[str] = field(default_factory=list)

    def add(self, detail: str) -> None:
        """Record a resolution step. Empty strings are ignored."""
        if detail:
            self.details.append(detail)


@dataclass
class ResolutionResult:
    """Resolution outcome. State changes are already applied to GameCore,
    so this only carries the details from ResolveDetails.
    """

    details: list[str]


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
        """Return resolution priority (10 or 11)."""

    @abstractmethod
    def resolve(self, game_core: "GameCore", details: ResolveDetails) -> None:
        """Execute effect resolution, directly modifying GameCore player state.

        Reads resolution context via game_core.get_resolve_* methods.
        Records settlement details via details.add().
        """

    # ------------------------------------------------------------------
    # Helper — resolve player id to Player instance
    # ------------------------------------------------------------------

    def _source_player(self, gc: "GameCore") -> "Player":
        """Return the source player (the one who played the card)."""
        return gc.player_1 if gc.get_resolve_source() == "1" else gc.player_2

    def _opponent_player(self, gc: "GameCore") -> "Player":
        """Return the opponent player."""
        return gc.player_2 if gc.get_resolve_source() == "1" else gc.player_1


# =============================================================================
# Concrete Resolver subclasses — one per card effect
# =============================================================================


class AttackDamageResolver(Resolver):
    """Settles ATTACK damage (priority 10).

    Damage = min(streak + 1, 3). Blocked by opponent DEFENSE status.
    """

    # Maximum damage from consecutive attack streak
    MAX_STREAK_DAMAGE = 3

    def get_priority(self) -> int:
        return 10

    def resolve(self, game_core: "GameCore", details: ResolveDetails) -> None:
        src_id = game_core.get_resolve_source()
        opponent = self._opponent_player(game_core)
        streak = game_core.get_resolve_attack_streak()
        raw_damage = min(streak + 1, self.MAX_STREAK_DAMAGE)

        if opponent.has_status(StatusType.DEFENSE):
            details.add(f"P{src_id} Attack(streak={streak}) blocked by opponent defense")
        else:
            opponent.take_damage(raw_damage)
            details.add(f"P{src_id} Attack(streak={streak}) deals {raw_damage} to opponent")


class GrantDefenseResolver(Resolver):
    """Grants DEFENSE status to the source player (priority 11)."""

    def get_priority(self) -> int:
        return 11

    def resolve(self, game_core: "GameCore", details: ResolveDetails) -> None:
        src_id = game_core.get_resolve_source()
        player = self._source_player(game_core)
        player.grant_status(StatusType.DEFENSE)
        details.add(f"P{src_id} gains DEFENSE")


class GrantGuardResolver(Resolver):
    """Grants GUARD status if source player had DEFENSE pre-round (priority 11).

    Does NOT grant GUARD if already present.
    """

    def get_priority(self) -> int:
        return 11

    def resolve(self, game_core: "GameCore", details: ResolveDetails) -> None:
        if not game_core.get_resolve_had_defense_pre():
            return

        src_id = game_core.get_resolve_source()
        player = self._source_player(game_core)

        # Check if GUARD already present (avoid duplicate)
        if player.has_status(StatusType.GUARD):
            return

        player.grant_status(StatusType.GUARD)
        details.add(f"P{src_id} gains GUARD (consecutive defense)")


class HeavyDamageResolver(Resolver):
    """Settles HEAVY strike damage (priority 10).

    Deals 1 damage, blocked by opponent GUARD status only (DEFENSE does not block).
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, game_core: "GameCore", details: ResolveDetails) -> None:
        src_id = game_core.get_resolve_source()
        opponent = self._opponent_player(game_core)

        if opponent.has_status(StatusType.GUARD):
            details.add(f"P{src_id} Heavy Strike blocked by opponent GUARD")
        else:
            opponent.take_damage(1)
            details.add(f"P{src_id} Heavy Strike deals 1 to opponent")


class ReflectResolver(Resolver):
    """Settles COUNTER reflect damage (priority 10).

    Reflects 1 damage if opponent played ATTACK or HEAVY.
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, game_core: "GameCore", details: ResolveDetails) -> None:
        opp_card = game_core.get_resolve_opponent_card()
        if opp_card not in (CardType.ATTACK, CardType.HEAVY):
            return

        src_id = game_core.get_resolve_source()
        opponent = self._opponent_player(game_core)
        opponent.take_damage(1)
        details.add(f"P{src_id} Counter reflects {opp_card.value}, deals 1 to opponent")


class GainEnergyResolver(Resolver):
    """Grants 1 energy to the source player (priority 10).

    Clamping to [0,5] is handled by Player.add_energy().
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, game_core: "GameCore", details: ResolveDetails) -> None:
        src_id = game_core.get_resolve_source()
        player = self._source_player(game_core)
        player.add_energy(1)
        details.add(f"P{src_id} gains 1 energy")


class FlexDamageResolver(Resolver):
    """Settles FLEX damage portion (priority 10).

    Deals 1 damage if opponent does NOT have DEFENSE status.
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, game_core: "GameCore", details: ResolveDetails) -> None:
        src_id = game_core.get_resolve_source()
        opponent = self._opponent_player(game_core)

        if opponent.has_status(StatusType.DEFENSE):
            details.add(f"P{src_id} FLEX damage blocked by opponent defense")
        else:
            opponent.take_damage(1)
            details.add(f"P{src_id} FLEX damage deals 1 to opponent")


class BuildResolver(Resolver):
    """No immediate effect — BUILD construction handled by GameCore (priority 10)."""

    def get_priority(self) -> int:
        return 10

    def resolve(self, game_core: "GameCore", details: ResolveDetails) -> None:
        src_id = game_core.get_resolve_source()
        details.add(f"P{src_id} BUILD — no immediate effect")


# =============================================================================
# CardResolverRegistry — CardType -> list[Resolver]
# =============================================================================


class CardResolverRegistry:
    """Registry mapping CardType to its Resolver list.

    All methods are classmethods. Registration happens at module load time.
    The registry is a class-level dict with default entries for all 7 CardTypes.
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

    Replaces the old _process_effect module-level function. Queries
    CardResolverRegistry for each card's resolvers, sorts by priority,
    sets resolution context on GameCore, and delegates to each Resolver.
    """

    @staticmethod
    def get_consecutive_attack(history: list[PlayerHistoryItem]) -> int:
        """Count consecutive ATTACK cards from the end of history.

        Only ATTACK cards count toward the streak. COUNTER, DEFENSE, and
        all other card types break it.

        Per design/16 D5, accepts list[PlayerHistoryItem] and reads
        item.card for the card type.

        Args:
            history: Chronological list of PlayerHistoryItem entries.

        Returns:
            Number of consecutive ATTACK cards at the end of history.
        """
        count = 0
        for item in reversed(history):
            if item.card == CardType.ATTACK:
                count += 1
            else:
                break
        return count

    @staticmethod
    def resolve(
        game_core: "GameCore",
        cards_1: list[CardType],
        cards_2: list[CardType],
        p1_streak: int,
        p2_streak: int,
        p1_had_defense_pre: bool,
        p2_had_defense_pre: bool,
    ) -> ResolutionResult:
        """Orchestrate resolution of both players' cards.

        Flow:
        1. Collect all (card_type, source_player, streak, had_defense_pre)
        2. For each, query CardResolverRegistry for resolvers
        3. Flatten into (resolver, source, opponent_card, streak, had_defense) tuples
        4. Sort by resolver.get_priority() (11 before 10)
        5. Set GameCore resolve context and invoke each resolver

        Args:
            game_core: GameCore instance whose state will be modified.
            cards_1: Cards played by Player 1 (expects exactly 1).
            cards_2: Cards played by Player 2 (expects exactly 1).
            p1_streak: P1's consecutive attack count before this round.
            p2_streak: P2's consecutive attack count before this round.
            p1_had_defense_pre: Whether P1 had DEFENSE before this round.
            p2_had_defense_pre: Whether P2 had DEFENSE before this round.

        Returns:
            ResolutionResult with collected settlement details.
        """
        details = ResolveDetails()

        # Build resolution tasks: (resolver, source_id, opponent_card, streak, had_defense)
        tasks: list[tuple[Resolver, str, CardType, int, bool]] = []

        for card_type in cards_1:
            for resolver in CardResolverRegistry.get_resolvers(card_type):
                tasks.append((resolver, "1", cards_2[0], p1_streak, p1_had_defense_pre))

        for card_type in cards_2:
            for resolver in CardResolverRegistry.get_resolvers(card_type):
                tasks.append((resolver, "2", cards_1[0], p2_streak, p2_had_defense_pre))

        # Sort by priority: 11 before 10
        tasks.sort(key=lambda t: t[0].get_priority(), reverse=True)

        # Execute each resolver with its context
        for resolver, source_id, opp_card, streak, had_defense in tasks:
            # Set resolution context on GameCore before invoking resolver
            game_core._resolve_source = source_id
            game_core._resolve_opponent_card = opp_card
            game_core._resolve_attack_streak = streak
            game_core._resolve_had_defense_pre = had_defense
            resolver.resolve(game_core, details)

        return ResolutionResult(details=list(details.details))
