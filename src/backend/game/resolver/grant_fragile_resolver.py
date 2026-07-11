"""GrantFragileResolver — extends opponent's FRAGILE status by 2 rounds
(priority 10).

Per intention #36: fragile changed from "grant(lifetime=2)"
to additive "+2 rounds". Blocked by opponent GUARD.
"""

from src.backend.game.player_state import StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class GrantFragileResolver(Resolver):
    """Extends opponent's FRAGILE status by 2 rounds (priority 10).

    Per intention #36 D3: uses Player.extend_status() for additive
    fragile instead of grant_status() with fixed lifetime=2.
    """

    FRAGILE_EXTRA: int = 2

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Extend FRAGILE by 2 rounds unless GUARD blocks it.

        Per intention #36 D3: calls opponent.extend_status() to add
        2 rounds to existing FRAGILE, or grant new FRAGILE(2) if none.

        Args:
            ctx: ResolveContext with card_owner and opponent.
            details: Collects resolution detail strings.
        """
        if ctx.opponent.has_status(StatusType.GUARD):
            details.add("Fragile blocked by opponent GUARD")
            return
        ctx.opponent.extend_status(StatusType.FRAGILE, extra=self.FRAGILE_EXTRA)
        details.add(f"extends FRAGILE by {self.FRAGILE_EXTRA} rounds")
