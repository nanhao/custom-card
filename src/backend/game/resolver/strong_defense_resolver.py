"""StrongDefenseResolver — grants DEFENSE status with lifetime=2 (priority 11).

Per design/37: 严防死守 (IRON_DEFENSE) grants DEFENSE with 2-round
lifetime instead of the standard 1 round. Blocked by FRAGILE status.
"""

from src.backend.game.player_state import StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class StrongDefenseResolver(Resolver):
    """Grants DEFENSE status with lifetime=2 to card_owner (P11).

    Per design/37: used exclusively by 严防死守 (IRON_DEFENSE).
    Standard GrantDefenseResolver grants lifetime=1; this resolver
    grants lifetime=2 for stronger defense. Blocked by FRAGILE.
    """

    def get_priority(self) -> int:
        return 11

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Grant DEFENSE with lifetime=2, unless blocked by FRAGILE.

        Args:
            ctx: ResolveContext with card_owner.
            details: Collects resolution detail strings.
        """
        if ctx.card_owner.has_status(StatusType.FRAGILE):
            details.add("DEFENSE(lifetime=2) blocked by FRAGILE")
            return
        # B3 FIX: _grant_status creates a NEW independent DEFENSE(2).
        # Previously used extend_status() which would add 2 to existing
        # DEFENSE lifetime instead of creating a separate status.
        ctx.card_owner._grant_status(StatusType.DEFENSE, lifetime=2)
        details.add("gains DEFENSE (lifetime=2)")
