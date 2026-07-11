"""BonusDefenseResolver — grants +1 extra DEFENSE if last round played
a '防'-tagged card (priority 11).

Used exclusively by 攻守兼备 (FLEX) per intention #36 card.md:
"如果本轮及上轮有打出过'防'卡，则额外增加一回合防御状态".

Per D2: "本轮打出过" excludes the current card, so only checks
card_owner.history[-1] (last round).

Blocked by FRAGILE — same as GrantDefenseResolver.
"""

from src.backend.game.player_state import StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class BonusDefenseResolver(Resolver):
    """Grants +1 extra DEFENSE if last round played a '防'-tagged card (P11)."""

    def get_priority(self) -> int:
        return 11

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Grant +1 DEFENSE if last round had a '防' tagged card.

        Per D2: only checks history[-1], does NOT include the current card.

        Args:
            ctx: ResolveContext with card_owner and opponent.
            details: Collects resolution detail strings.
        """
        # First round — no history.
        if len(ctx.card_owner.history) == 0:
            return

        # Check if last round's card has '防' tag.
        last_card = ctx.card_owner.history[-1].card
        if last_card is None:
            return
        # Lazy import to avoid circular dependency (card_def → resolver).
        from src.backend.game.card_def import get_card_def  # noqa: PLC0415
        tags = get_card_def(last_card).tags
        if "防" not in tags:
            return

        # Blocked by FRAGILE (same as GrantDefenseResolver).
        if ctx.card_owner.has_status(StatusType.FRAGILE):
            details.add("Bonus DEFENSE blocked by FRAGILE")
            return

        ctx.card_owner.grant_status(StatusType.DEFENSE)
        details.add("gains +1 bonus DEFENSE")
