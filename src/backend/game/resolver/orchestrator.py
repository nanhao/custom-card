"""CardResolver — priority-layered resolution orchestrator.

Queries CardResolverRegistry for each card's resolvers, builds a
ResolveContext per card, sorts by priority, and delegates to each Resolver.
"""

from src.backend.game.card import CardType
from src.backend.game.player import Player, PlayerHistoryItem
from src.backend.game.resolver.base import (
    ResolveContext,
    ResolveDetails,
    ResolutionResult,
    Resolver,
)
from src.backend.game.resolver.registry import CardResolverRegistry


class CardResolver:
    """Priority-layered resolution orchestrator."""

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
        player_1: Player,
        player_2: Player,
        cards_1: list[CardType],
        cards_2: list[CardType],
    ) -> ResolutionResult:
        """Orchestrate resolution of both players' cards."""
        details = ResolveDetails()

        tasks: list[tuple[Resolver, ResolveContext]] = []

        for card_type in cards_1:
            ctx = ResolveContext(
                card_owner=player_1,
                opponent=player_2,
                card_played_by_owner=card_type,
                card_played_by_opponent=cards_2[0],
            )
            for resolver in CardResolverRegistry.get_resolvers(card_type):
                tasks.append((resolver, ctx))

        for card_type in cards_2:
            ctx = ResolveContext(
                card_owner=player_2,
                opponent=player_1,
                card_played_by_owner=card_type,
                card_played_by_opponent=cards_1[0],
            )
            for resolver in CardResolverRegistry.get_resolvers(card_type):
                tasks.append((resolver, ctx))

        tasks.sort(key=lambda t: t[0].get_priority(), reverse=True)

        for resolver, ctx in tasks:
            resolver.resolve(ctx, details)

        return ResolutionResult(details=list(details.details))
