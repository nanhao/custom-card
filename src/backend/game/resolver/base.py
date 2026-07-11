"""Base types for the resolver strategy pattern (design/21).

Defines ResolveDetails, ResolutionResult, ResolveContext, and the
Resolver abstract base class that all effect resolvers implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.backend.game.card_type import CardType
from src.backend.game.player import Player
from src.backend.game.player_state import StatusType


@dataclass
class ResolveDetails:
    """Settlement detail collector, the data source for ResolutionResult.details."""

    details: list[str] = field(default_factory=list)

    def add(self, detail: str) -> None:
        """Record a resolution step. Empty strings are ignored."""
        if detail:
            self.details.append(detail)


@dataclass
class ResolutionResult:
    """Resolution outcome. State changes are already applied to GameCore players,
    so this only carries the details from ResolveDetails.
    """

    details: list[str]


@dataclass
class ResolveContext:
    """All context needed by a Resolver to settle one card's effects.

    Passed directly to resolve() — no GameCore indirection needed.
    Per design/25: build_target and build_target_cost added for BUILD.
    Per design/37: dismantle_target added for BUILD's dismantle mode.
    """

    card_owner: Player
    opponent: Player
    card_played_by_owner: CardType
    card_played_by_opponent: CardType | None  # None when opponent passes (design/28)
    build_target: CardType | None = None
    build_target_cost: int = 0
    dismantle_target: CardType | None = None  # design/37: BUILD dismantle mode


class Resolver(ABC):
    """Abstract base for card effect resolvers.

    Each subclass encapsulates a single effect resolution logic. Subclasses
    are stateless singletons registered via CardResolverRegistry.
    """

    @abstractmethod
    def get_priority(self) -> int:
        """Return resolution priority."""

    @abstractmethod
    def resolve(self, ctx: ResolveContext, details: ResolveDetails) -> None:
        """Execute effect resolution, directly modifying Player state via ctx.

        Records settlement details via details.add().
        """
