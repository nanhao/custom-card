"""Read-only player state snapshot for the custom card game v2 (design/34).

Provides PlayerStateSnapshot dataclass — a pure data container representing
one player's state at a point in time. Per design/19 D1, fields are typed
(CardType, PlayerStatus) — no strings or dicts.

Per design/34: draw_pile/discard_pile are always list[CardType] (never None),
since both players' piles are mutually visible.

Per architecture overview §2: game layer is pure logic — no UI, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.backend.game.card_type import CardType
from src.backend.game.player_history_item import PlayerHistoryItem
from src.backend.game.player_state import PlayerStatus


@dataclass
class PlayerStateSnapshot:
    """Read-only snapshot of a single player's state at a point in time.

    Attributes:
        hp: Current HP (≥ 0).
        energy: Current energy (0–5).
        statuses: Active statuses with lifetimes (PlayerStatus objects, not dicts).
        hand: Cards currently in hand (CardType enum, not strings).
        history: Full action history (card + build_target per round).
        draw_pile: Cards in draw pile (design/34: always visible to both players).
        discard_pile: Cards in discard pile (design/34: always visible to both players).
        draw_pile_count: Count of cards in draw pile (always visible).
        discard_pile_count: Count of cards in discard pile (always visible).
    """

    hp: int
    energy: int
    statuses: list[PlayerStatus]
    hand: list[CardType]
    history: list[PlayerHistoryItem]
    draw_pile: list[CardType] = field(default_factory=list)
    discard_pile: list[CardType] = field(default_factory=list)
    draw_pile_count: int = 0
    discard_pile_count: int = 0

    def to_dict(self) -> dict:
        """Serialize this PlayerStateSnapshot to a JSON-safe dict.

        Converts CardType → string via .value, PlayerStatus → dict via
        .to_dict(), PlayerHistoryItem → {"card": str, "build_target":
        str|None, "dismantle_target": str|None}.
        """
        result: dict = {
            "hp": self.hp,
            "energy": self.energy,
            "statuses": [st.to_dict() for st in self.statuses],
            "hand": [c.value for c in self.hand],
            "history": [
                {
                    "card": h.card.value if h.card else None,
                    "build_target": h.build_target.value if h.build_target else None,
                    "dismantle_target": h.dismantle_target.value if h.dismantle_target else None,
                }
                for h in self.history
            ],
            "draw_pile_count": self.draw_pile_count,
            "discard_pile_count": self.discard_pile_count,
        }
        # Per design/34: draw_pile and discard_pile are always serialized
        # since both players' piles are mutually visible.
        if self.draw_pile is not None:
            result["draw_pile"] = [c.value for c in self.draw_pile]
        if self.discard_pile is not None:
            result["discard_pile"] = [c.value for c in self.discard_pile]
        return result
