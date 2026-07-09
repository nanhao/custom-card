"""GainEnergyResolver — grants 1 energy to the card_owner (priority 10).

Clamping to [0,5] is handled by Player.add_energy().
"""

from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class GainEnergyResolver(Resolver):
    """Grants 1 energy to the card_owner (priority 10)."""

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        ctx.card_owner.add_energy(1)
        details.add("gains 1 energy")
