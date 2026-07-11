"""GrantGuardResolver — grants GUARD if card_owner played a defense-granting
card last round (priority 12).

Per card.md (intention/card.md): DEFENSE effect 2 — "如果己方上回合有防御状态，
则赋予己方联防状态"。Since DEFENSE has lifetime=1 and is removed by the round-end
status tick, we detect it by checking if the previous round's played card was
a defense-granting type (DEFENSE, COUNTER, or FLEX).

Priority 12 runs BEFORE GrantDefenseResolver (priority 11).
Per design/24: FRAGILE blocks gaining GUARD.
"""

from src.backend.game.card_type import CardType
from src.backend.game.player_state import StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver

# Cards whose effect 1 grants DEFENSE to caster (per card.md).
_DEFENSE_GRANTING_CARDS = {CardType.DEFENSE, CardType.COUNTER, CardType.FLEX}


class GrantGuardResolver(Resolver):
    """Grants GUARD if card_owner played defense-granting card last round.

    Per card.md: Defense's second effect triggers when the player had
    defense status in the previous round. Since defense expires after
    each round's tick, we detect this via the player's action history.
    """

    def get_priority(self) -> int:
        return 12

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        # Check if card_owner played a defense-granting card last round.
        if len(ctx.card_owner.history) == 0:
            return
        last_card = ctx.card_owner.history[-1].card
        if last_card not in _DEFENSE_GRANTING_CARDS:
            return
        # Per design/24: FRAGILE blocks GUARD.
        if ctx.card_owner.has_status(StatusType.FRAGILE):
            details.add("GUARD blocked by FRAGILE")
            return
        if ctx.card_owner.has_status(StatusType.GUARD):
            return
        ctx.card_owner.grant_status(StatusType.GUARD)
        details.add("gains GUARD (played defense last round)")
