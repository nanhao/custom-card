"""GrantDefenseResolver — grants DEFENSE status to the card_owner (priority 11)."""

from src.backend.game.player_state import StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class GrantDefenseResolver(Resolver):
    """Grants DEFENSE status to the card_owner (priority 11)."""

    def get_priority(self) -> int:
        return 11

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        ctx.card_owner.grant_status(StatusType.DEFENSE)
        details.add("gains DEFENSE")
