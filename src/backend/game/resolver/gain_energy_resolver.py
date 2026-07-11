"""GainEnergyResolver — grants energy to card_owner (priority 10).

Per intention #36: baseline 1 energy + bonus 1 if ENERGY was played
in "本轮及上轮" (current cycle before this round + previous cycle).

Clamping to [0,5] is handled by Player.add_energy().
"""

from src.backend.game.card_type import CardType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class GainEnergyResolver(Resolver):
    """Grants energy to card_owner (priority 10).

    Per intention #36: 1 energy baseline; +1 bonus if ENERGY was
    played in 本轮(current cycle before this round) or 上轮(previous cycle).
    Per D2: the current card itself is excluded.
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Grant 1 energy, or 2 if ENERGY was played in 本轮/上轮.

        Formula per card.md:
          j = i - (i % ROUNDS_PER_CYCLE)
          check history[max(0, j - ROUNDS_PER_CYCLE) : i]

        Args:
            ctx: ResolveContext with card_owner and opponent.
            details: Collects resolution detail strings.
        """
        # Baseline: always +1 energy.
        ctx.card_owner.add_energy(1)

        history = ctx.card_owner.history
        i = len(history)
        if i > 0:
            # Lazy import to avoid circular dependency (card_def → resolver → game_core).
            from src.backend.game.constants import ROUNDS_PER_CYCLE  # noqa: PLC0415
            j = i - (i % ROUNDS_PER_CYCLE)
            start = max(0, j - ROUNDS_PER_CYCLE)
            bonus = any(h.card == CardType.ENERGY for h in history[start:i])
        else:
            bonus = False

        if bonus:
            ctx.card_owner.add_energy(1)
            details.add("gains 2 energy (consecutive)")
        else:
            details.add("gains 1 energy")
