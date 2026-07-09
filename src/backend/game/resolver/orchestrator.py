"""CardResolver — priority-layered resolution orchestrator.

Per design/22: CardResolver.resolve() accepts pre-built (Resolver, ResolveContext)
tasks from the caller. The caller (GameCore.play_round) is responsible for
building tasks by querying get_card_def(card_type).resolvers.

This avoids requiring CardResolver to import card_def, eliminating the
circular dependency without lazy imports.
"""

from src.backend.game.card_type import CardType
from src.backend.game.player import PlayerHistoryItem
from src.backend.game.resolver.base import (
    ResolveContext,
    ResolveDetails,
    ResolutionResult,
    Resolver,
)


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
        tasks: list[tuple[Resolver, ResolveContext]],
    ) -> ResolutionResult:
        """Orchestrate resolution of pre-built (resolver, context) tasks.

        Per design/22: tasks are built by the caller (GameCore.play_round)
        using get_card_def(card_type).resolvers. CardResolver only sorts
        by priority and delegates execution.

        Args:
            tasks: List of (Resolver instance, ResolveContext) pairs
                   built by the caller from CardDef.resolvers.

        Returns:
            ResolutionResult containing all resolution detail strings.
        """
        details = ResolveDetails()

        # Sort by priority descending: 12 > 11 > 10
        tasks.sort(key=lambda t: t[0].get_priority(), reverse=True)

        for resolver, ctx in tasks:
            resolver.resolve(ctx, details)

        return ResolutionResult(details=list(details.details))
