"""ReflectResolver — settles COUNTER reflect damage (priority 10).

Reflects 1 damage if opponent played ATTACK or HEAVY.
"""

from src.backend.game.card_type import CardType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class ReflectResolver(Resolver):
    """Settles COUNTER reflect damage (priority 10)."""

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        if ctx.card_played_by_opponent not in (CardType.ATTACK, CardType.HEAVY):
            return
        ctx.opponent.take_damage(1)
        details.add(
            f"Counter reflects {ctx.card_played_by_opponent.value}, deals 1 to opponent"
        )
