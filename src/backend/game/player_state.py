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
