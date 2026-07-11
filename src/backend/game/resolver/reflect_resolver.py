"""ReflectResolver — settles COUNTER reflect damage (priority 10).

Per intention #36: condition changed from "opponent played ATTACK or HEAVY"
to "opponent's card has 攻 tag AND opponent has no DEFENSE status".
"""

from src.backend.game.player_state import StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class ReflectResolver(Resolver):
    """Settles COUNTER reflect damage (priority 10).

    Per intention #36 D4: checks opponent card's tags for "攻" instead
    of hardcoding ATTACK/HEAVY card types. Also requires opponent to
    have no DEFENSE status.
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Deal 1 reflect damage if opponent's card has '攻' tag
        and opponent has no DEFENSE.

        Args:
            ctx: ResolveContext with card_owner and opponent.
            details: Collects resolution detail strings.
        """
        # Opponent passed — no card to check.
        if ctx.card_played_by_opponent is None:
            return
        # Lazy import to avoid circular dependency (card_def → resolver).
        from src.backend.game.card_def import get_card_def  # noqa: PLC0415
        # Check if opponent's card has '攻' tag.
        opponent_tags = get_card_def(ctx.card_played_by_opponent).tags
        if "攻" not in opponent_tags:
            return
        # Opponent has DEFENSE — reflect blocked.
        if ctx.opponent.has_status(StatusType.DEFENSE):
            details.add("Counter reflect blocked by opponent DEFENSE")
            return
        ctx.opponent.take_damage(1)
        details.add(
            f"Counter reflects {ctx.card_played_by_opponent.value}, deals 1 to opponent"
        )
