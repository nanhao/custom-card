"""GrantGuardResolver — grants GUARD if card_owner played a card with
'防' tag last round (priority 12).

Per intention #36: guard trigger condition is "上回合有防御状态或联防状态".
Since DEFENSE has lifetime=1 and is removed by the round-end status tick,
we detect it by checking if last round's card had '防' tag — all '防'-tagged
cards grant DEFENSE, which is a reliable proxy.

Priority 12 runs BEFORE GrantDefenseResolver (priority 11).
Per design/24: FRAGILE blocks gaining GUARD.
"""

from src.backend.game.player_state import StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class GrantGuardResolver(Resolver):
    """Grants GUARD if card_owner played a '防'-tagged card last round.

    Per intention #36 D4: uses tag-based check, consistent with
    ReflectResolver and BonusDefenseResolver.
    """

    def get_priority(self) -> int:
        return 12

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Grant GUARD if last round's card has '防' tag.

        Args:
            ctx: ResolveContext with card_owner and opponent.
            details: Collects resolution detail strings.
        """
        # First round — no history.
        if len(ctx.card_owner.history) == 0:
            return
        last_card = ctx.card_owner.history[-1].card
        if last_card is None:
            return
        # Lazy import to avoid circular dependency (card_def → resolver).
        from src.backend.game.card_def import get_card_def  # noqa: PLC0415
        if "防" not in get_card_def(last_card).tags:
            return
        # Per design/24: FRAGILE blocks GUARD.
        if ctx.card_owner.has_status(StatusType.FRAGILE):
            details.add("GUARD blocked by FRAGILE")
            return
        ctx.card_owner.extend_status(StatusType.GUARD, 1)
        details.add("gains GUARD (played defense last round)")
