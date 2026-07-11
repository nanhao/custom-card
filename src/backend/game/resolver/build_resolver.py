"""BuildResolver — handles BUILD card construction (priority 10).

Per design/25: BUILD settlement logic moved from GameCore._process_build()
to here, unifying all card effect resolution under the Resolver system."""

from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class BuildResolver(Resolver):
    """Executes BUILD card effect: deduct energy and add built card to draw_pile.

    Per design/25: the resolve() method reads ctx.build_target to determine
    which card the player chose to construct, and ctx.build_target_cost
    for the energy cost. It deducts the cost from card_owner's energy and
    appends the target card to draw_pile.

    Per design/37: card goes to draw_pile (not discard_pile).
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Execute BUILD settlement: deduct energy, add card to draw_pile.

        Defensive: if ctx.build_target is None, returns immediately with
        no state changes (should not happen in normal flow; _validate
        ensures build_target is set before play_round proceeds).
        """
        if ctx.build_target is None:
            return
        cost: int = ctx.build_target_cost
        ctx.card_owner.energy = max(0, ctx.card_owner.energy - cost)
        ctx.card_owner.draw_pile.append(ctx.build_target)
        details.add(f"BUILD {ctx.build_target.value} -> draw pile (cost {cost})")
