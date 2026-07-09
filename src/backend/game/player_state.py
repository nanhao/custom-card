"""Player status type definitions for the custom card game (design/15).

Defines StatusType enum for all possible statuses and PlayerStatus dataclass
that bundles a status type with its remaining lifetime.

Per architecture D7, this module is separate from player.py to keep status
type definitions focused and avoid circular imports.
"""

from dataclasses import dataclass
from enum import Enum


class StatusType(Enum):
    """All possible player status types in the game.

    Values are serialized as their string representation for JSON/API output.
    """

    DEFENSE = "defense"  # 防御状态 — blocks attack damage
    GUARD = "guard"      # 联防状态 — blocks heavy strike damage
    FRAGILE = "fragile"  # 脆弱状态 (design/24) — 持有者无法获得防御和联防效果


@dataclass
class PlayerStatus:
    """Single status instance on a player, with remaining lifetime.

    Each round, lifetime is decremented by one. When lifetime reaches 0,
    the status is removed during _tick_status_lifetimes().

    Attributes:
        status_type: The type of status (DEFENSE or GUARD).
        lifetime: Remaining rounds before expiry. > 0 means active.
    """

    status_type: StatusType
    lifetime: int

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict: {"type": "defense", "lifetime": 1}.

        Used by get_state(), session_store, and serializers to produce
        a consistent serialization format.
        """
        return {"type": self.status_type.value, "lifetime": self.lifetime}

    def __repr__(self) -> str:
        """Compact repr: PlayerStatus(DEFENSE, lifetime=1)."""
        return f"PlayerStatus({self.status_type.name}, lifetime={self.lifetime})"


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def has_active_status(
    statuses: list[PlayerStatus], status_type: StatusType
) -> bool:
    """Return True if any status in the list matches type and has lifetime > 0.

    Shared by Player.has_status() and CardResolver._resolve()'s pre-existing
    status checks (design/15 §3.2, §6.1).
    """
    return any(
        ps.status_type == status_type and ps.lifetime > 0
        for ps in statuses
    )
