"""GrantDefenseResolver — grants DEFENSE status to the card_owner (priority 11).

Per design/24: 拥有脆弱状态的玩家无法获得防御效果。
"""

from src.backend.game.player_state import StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class GrantDefenseResolver(Resolver):
    """Grants DEFENSE status to the card_owner (priority 11).

    Per design/24: FRAGILE status blocks gaining DEFENSE.
    """

    def get_priority(self) -> int:
        return 11

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        if ctx.card_owner.has_status(StatusType.FRAGILE):
            details.add("DEFENSE blocked by FRAGILE")
            return
        ctx.card_owner.grant_status(StatusType.DEFENSE)
        details.add("gains DEFENSE")
