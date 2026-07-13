"""Player action history entry for the custom card game v2 (design/14 D8, design/37).

Per architecture D2, PlayerHistoryItem is a pure data container (dataclass)
recording the card played, build target, and dismantle target per round.

Per architecture D9, this module is pure logic — no UI, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.backend.game.card_type import CardType


@dataclass
class PlayerHistoryItem:
    """Single entry in a player's action history (design/14 D8, design/37).

    Records the card played, the build target (if build mode), and the
    dismantle target (if dismantle mode) for a single round. Used by
    GameCore.get_state() to provide complete action history.

    Per design/37: dismantle_target field added for BUILD card's
    dismantle mode (lower half of 构建/拆解).

    Attributes:
        card: The card played this round.
        build_target: The card built via BUILD, or None.
        dismantle_target: The card dismantled via BUILD, or None.
    """

    card: CardType | None  # None means pass (design/28)
    build_target: CardType | None = None
    dismantle_target: CardType | None = None  # design/37
