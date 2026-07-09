"""FlexDamageResolver — settles FLEX damage portion (priority 10).

Deals 1 damage if opponent does NOT have DEFENSE status.
"""

from src.backend.game.player_state import StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class FlexDamageResolver(Resolver):
    """Settles FLEX damage portion (priority 10)."""

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        if ctx.opponent.has_status(StatusType.DEFENSE):
            details.add("FLEX damage blocked by opponent defense")
        else:
            ctx.opponent.take_damage(1)
            details.add("FLEX damage deals 1 to opponent")
