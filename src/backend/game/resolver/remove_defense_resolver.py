"""RemoveDefenseResolver — removes opponent DEFENSE (priority 10).

Used exclusively by 破防重击 (HEAVY) per intention #36 card.md:
"立刻移除对方的防御状态".

Unlike HeavyDamageResolver and GrantFragileResolver, this resolver
is NOT blocked by GUARD — GUARD only blocks damage and fragile.
"""

from src.backend.game.player_state import StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class RemoveDefenseResolver(Resolver):
    """Removes DEFENSE status from opponent (priority 10)."""

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Remove opponent's DEFENSE if present.

        Per intention #36: GUARD does NOT block defense removal — only
        damage and fragile are blocked by GUARD.

        Args:
            ctx: ResolveContext with card_owner and opponent.
            details: Collects resolution detail strings.
        """
        if ctx.opponent.has_status(StatusType.DEFENSE):
            ctx.opponent.remove_status(StatusType.DEFENSE)
            details.add("removes opponent DEFENSE")
