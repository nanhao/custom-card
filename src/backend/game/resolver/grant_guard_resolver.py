"""GrantGuardResolver — grants GUARD if card_owner had DEFENSE pre-round (priority 12).

Priority 12 runs BEFORE any GrantDefenseResolver (priority 11), so
card_owner.has_status(DEFENSE) reflects the pre-round state.
Does NOT grant GUARD if already present or if card_owner has FRAGILE.
Per design/24: 拥有脆弱状态的玩家无法获得联防效果。
"""

from src.backend.game.player_state import StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class GrantGuardResolver(Resolver):
    """Grants GUARD if card_owner had DEFENSE pre-round (priority 12).

    Per design/24: FRAGILE status blocks gaining GUARD.
    """

    def get_priority(self) -> int:
        return 12

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        if ctx.card_owner.has_status(StatusType.FRAGILE):
            details.add("GUARD blocked by FRAGILE")
            return
        if not ctx.card_owner.has_status(StatusType.DEFENSE):
            return
        if ctx.card_owner.has_status(StatusType.GUARD):
            return
        ctx.card_owner.grant_status(StatusType.GUARD)
        details.add("gains GUARD (consecutive defense)")
