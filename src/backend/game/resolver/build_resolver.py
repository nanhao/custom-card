"""BuildResolver — handles BUILD card construction (priority 10).

Per design/25: BUILD settlement logic moved from GameCore._process_build()
to here, unifying all card effect resolution under the Resolver system."""

from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class BuildResolver(Resolver):
    """Executes BUILD card effect: deduct energy and add built card to discard.

    Per design/25: the resolve() method reads ctx.build_target to determine
    which card the player chose to construct. It deducts the target card's
    build_cost from the card_owner's energy and appends the target card to
    the card_owner's discard_pile.
    """

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Execute BUILD settlement: deduct energy, add card to discard.

        Defensive: if ctx.build_target is None, returns immediately with
        no state changes (should not happen in normal flow; _validate
        ensures build_target is set before play_round proceeds).
        """
        if ctx.build_target is None:
            return
        cost: int = ctx.build_target.get_build_cost()
        ctx.card_owner.energy = max(0, ctx.card_owner.energy - cost)
        ctx.card_owner.discard_pile.append(ctx.build_target)
        details.add(f"BUILD {ctx.build_target.value} (cost {cost})")
