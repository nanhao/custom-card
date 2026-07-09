"""BuildResolver — no immediate effect, BUILD construction handled by GameCore (priority 10)."""

from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class BuildResolver(Resolver):
    """No immediate effect — BUILD construction handled by GameCore (priority 10)."""

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        details.add("BUILD — no immediate effect")
