"""GrantGuardResolver — grants GUARD if card_owner had DEFENSE pre-round (priority 12).

Priority 12 runs BEFORE any GrantDefenseResolver (priority 11), so
card_owner.has_status(DEFENSE) reflects the pre-round state.
Does NOT grant GUARD if already present.
"""

from src.backend.game.player_state import StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class GrantGuardResolver(Resolver):
    """Grants GUARD if card_owner had DEFENSE pre-round (priority 12)."""

    def get_priority(self) -> int:
        return 12

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        if not ctx.card_owner.has_status(StatusType.DEFENSE):
            return
        if ctx.card_owner.has_status(StatusType.GUARD):
            return
        ctx.card_owner.grant_status(StatusType.GUARD)
        details.add("gains GUARD (consecutive defense)")
