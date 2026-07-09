"""GrantFragileResolver — grants FRAGILE status to opponent (priority 10).

Per card.md (design/24): 重击赋予对方持续2回合的脆弱状态。
Blocked by opponent GUARD status — shared condition with HeavyDamageResolver.
"""

from src.backend.game.player_state import StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class GrantFragileResolver(Resolver):
    """Grants FRAGILE (lifetime=2) to opponent if not blocked by GUARD (priority 10).

    Per design/24 D2: FRAGILE is blocked by GUARD, same as the damage
    portion of Heavy Strike. Both effects share the same guard condition.
    """

    FRAGILE_LIFETIME: int = 2

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Grant FRAGILE (lifetime=2) to opponent unless GUARD blocks it.

        Args:
            ctx: ResolveContext with card_owner and opponent.
            details: Collects resolution detail strings.
        """
        if ctx.opponent.has_status(StatusType.GUARD):
            details.add("Fragile blocked by opponent GUARD")
            return
        ctx.opponent.grant_status(StatusType.FRAGILE, lifetime=self.FRAGILE_LIFETIME)
        details.add(f"gains FRAGILE ({self.FRAGILE_LIFETIME} rounds)")
