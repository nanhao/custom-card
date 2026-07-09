"""AttackDamageResolver — settles ATTACK damage (priority 10).

Damage = min(streak + 1, 3), where streak is computed from
card_owner's history via CardResolver.get_consecutive_attack().
Blocked by opponent DEFENSE status.
"""

from src.backend.game.player_state import StatusType
from src.backend.game.resolver.base import ResolveContext, ResolveDetails, Resolver


class AttackDamageResolver(Resolver):
    """Settles ATTACK damage (priority 10)."""

    MAX_STREAK_DAMAGE = 3

    def get_priority(self) -> int:
        return 10

    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        # Local import avoids circular dependency with orchestrator
        from src.backend.game.resolver.orchestrator import CardResolver

        streak = CardResolver.get_consecutive_attack(ctx.card_owner.history)
        raw_damage = min(streak + 1, self.MAX_STREAK_DAMAGE)

        if ctx.opponent.has_status(StatusType.DEFENSE):
            details.add(f"Attack(streak={streak}) blocked by opponent defense")
        else:
            ctx.opponent.take_damage(raw_damage)
            details.add(f"Attack(streak={streak}) deals {raw_damage} to opponent")
