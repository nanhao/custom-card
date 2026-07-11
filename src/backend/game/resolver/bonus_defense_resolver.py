"""BonusDefenseResolver — grants +1 extra DEFENSE if a '防'-tagged card
was played in "本轮及上轮" (priority 11).

Used exclusively by 攻守兼备 (FLEX) per intention #36 card.md:
"如果本轮及上轮有打出过'防'卡，则额外增加一回合防御状态".

Per D2: the current card itself is excluded.
Blocked by FRAGILE — same as GrantDefenseResolver.
"""

from src.backend.game.player_state import StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class BonusDefenseResolver(Resolver):
    """Grants +1 extra DEFENSE if '防'-tagged card was played in 本轮/上轮 (P11)."""

    def get_priority(self) -> int:
        return 11

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Grant +1 DEFENSE if 本轮/上轮 has a '防' tagged card.

        Formula per card.md:
          j = i - (i % ROUNDS_PER_CYCLE)
          check history[max(0, j - ROUNDS_PER_CYCLE) : i]

        Args:
            ctx: ResolveContext with card_owner and opponent.
            details: Collects resolution detail strings.
        """
        history = ctx.card_owner.history
        i = len(history)
        if i == 0:
            return

        # Lazy imports to avoid circular dependency (card_def → resolver → game_core).
        from src.backend.game.constants import ROUNDS_PER_CYCLE  # noqa: PLC0415
        from src.backend.game.card_def import get_card_def  # noqa: PLC0415

        j = i - (i % ROUNDS_PER_CYCLE)
        start = max(0, j - ROUNDS_PER_CYCLE)

        found = False
        for h in history[start:i]:
            if h.card is not None and "防" in get_card_def(h.card).tags:
                found = True
                break

        if not found:
            return

        # Blocked by FRAGILE (same as GrantDefenseResolver).
        if ctx.card_owner.has_status(StatusType.FRAGILE):
            details.add("Bonus DEFENSE blocked by FRAGILE")
            return

        ctx.card_owner.extend_status(StatusType.DEFENSE, 1)
        details.add("gains +1 bonus DEFENSE")
