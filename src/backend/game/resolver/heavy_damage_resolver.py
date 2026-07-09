"""HeavyDamageResolver — settles HEAVY strike damage (priority 10).

Deals 1 damage, blocked by opponent GUARD status only.
"""

from src.backend.game.player_state import StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class HeavyDamageResolver(Resolver):
    """Settles HEAVY strike damage (priority 10)."""

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        if ctx.opponent.has_status(StatusType.GUARD):
            details.add("Heavy Strike blocked by opponent GUARD")
        else:
            ctx.opponent.take_damage(1)
            details.add("Heavy Strike deals 1 to opponent")
