"""GainEnergyResolver — grants energy to card_owner (priority 10).

Per intention #36: baseline 1 energy + bonus 1 if ENERGY was played
last round (consecutive bonus).

Clamping to [0,5] is handled by Player.add_energy().
"""

from src.backend.game.card_type import CardType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class GainEnergyResolver(Resolver):
    """Grants energy to card_owner (priority 10).

    Per intention #36: 1 energy baseline; +1 bonus if ENERGY was
    played last round. Per D2, "本轮打出过聚能" excludes the current
    card, so only history[-1] is checked.
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Grant 1 energy, or 2 if consecutive ENERGY.

        Per D2: checks card_owner.history[-1] for ENERGY, which
        represents last round's card (not current).

        Args:
            ctx: ResolveContext with card_owner and opponent.
            details: Collects resolution detail strings.
        """
        # Baseline: always +1 energy.
        ctx.card_owner.add_energy(1)

        # Bonus: +1 if ENERGY was played last round.
        bonus = (
            len(ctx.card_owner.history) > 0
            and ctx.card_owner.history[-1].card == CardType.ENERGY
        )
        if bonus:
            ctx.card_owner.add_energy(1)
            details.add("gains 2 energy (consecutive)")
        else:
            details.add("gains 1 energy")
