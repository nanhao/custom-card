"""DismantleResolver — removes a card from discard pile (priority 10).

Per design/37: BUILD card gains a dismantle (拆解) mode. When the player
chooses the lower half of 构建/拆解, this resolver removes the selected
card from the player's discard pile, permanently removing it from the game.
No energy is consumed for dismantle.
"""

from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class DismantleResolver(Resolver):
    """Removes dismantle_target from card_owner's discard pile (P10).

    Per design/37: the dismantle mode of 构建/拆解 allows the player to
    remove a card from their discard pile at no energy cost. This resolver
    is registered alongside BuildResolver in CardDef for CardType.BUILD.
    Both resolvers are invoked during resolution — only the one whose
    target is non-None performs work; the other is a no-op.
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Remove dismantle_target from card_owner's discard pile.

        If ctx.dismantle_target is None (build mode was chosen instead),
        this resolver is a no-op.

        Args:
            ctx: ResolveContext with card_owner and dismantle_target.
            details: Collects resolution detail strings.
        """
        if ctx.dismantle_target is None:
            return
        ctx.card_owner.remove_from_discard(ctx.dismantle_target)
        details.add(f"DISMANTLE {ctx.dismantle_target.value}")
