"""Read-only game state snapshots for the custom card game v2 (design/19).

Provides GameState and PlayerState dataclasses that replace the dict returned
by GameCore.get_state(). Per design/19 D1, these are pure data containers
with typed fields — no strings for cards or statuses.

# Per architecture overview §2: game layer is pure logic — no UI, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.backend.game.card_type import CardType
from src.backend.game.player import PlayerHistoryItem
from src.backend.game.player_state import PlayerStatus

if TYPE_CHECKING:
    from src.backend.game.game_core import RoundResult


@dataclass
class PlayerState:
    """Read-only snapshot of a single player's state at a point in time.

    Attributes:
        hp: Current HP (≥ 0).
        energy: Current energy (0–5).
        statuses: Active statuses with lifetimes (PlayerStatus objects, not dicts).
        hand: Cards currently in hand (CardType enum, not strings).
        history: Full action history (card + build_target per round).
        draw_pile: Cards in draw pile, or None if hidden from this perspective.
        discard_pile: Cards in discard pile, or None if hidden.
        draw_pile_count: Count of cards in draw pile (always visible).
        discard_pile_count: Count of cards in discard pile (always visible).
    """

    hp: int
    energy: int
    statuses: list[PlayerStatus]
    hand: list[CardType]
    history: list[PlayerHistoryItem]
    draw_pile: list[CardType] | None = None
    discard_pile: list[CardType] | None = None
    draw_pile_count: int = 0
    discard_pile_count: int = 0


@dataclass
class GameState:
    """Read-only snapshot of the full game state.

    Attributes:
        round: Current round number (1-based).
        cycle: Current cycle number (1-based).
        round_in_cycle: Round position within current cycle (1–3, design/26 D1).
        players: Player states keyed by "1" and "2".
        is_over: True if game has ended.
        winner: "1", "2", "draw", or None if game ongoing.
        mode: "pvp" or "pve".
        round_log: Complete round result history.
    """

    round: int
    cycle: int
    round_in_cycle: int
    players: dict[str, PlayerState]
    is_over: bool
    winner: str | None
    mode: str
    round_log: list[RoundResult]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def to_dict(gs: GameState) -> dict:
        """Serialize a GameState to a JSON-safe dict.

        Converts CardType → string via .value, PlayerStatus → dict via
        .to_dict(), PlayerHistoryItem → {"card": str, "build_target": str|None},
        RoundResult → dict via RoundResult.to_dict().

        Per design/19 D4: this is the serialization boundary — Converter and
        AppManager consume the typed GameState directly.

        Args:
            gs: The GameState value object to serialize.

        Returns:
            dict with nested players and round_log structures,
            matching the existing JSON API schema.
        """
        # Lazy import to avoid circular dependency with game_core.py.
        from src.backend.game.game_core import RoundResult  # noqa: F811

        def _serialize_player(ps: PlayerState) -> dict:
            """Serialize a single PlayerState to a JSON-safe dict."""
            result: dict = {
                "hp": ps.hp,
                "energy": ps.energy,
                "statuses": [st.to_dict() for st in ps.statuses],
                "hand": [c.value for c in ps.hand],
                "history": [
                    {
                        "card": h.card.value,
                        "build_target": h.build_target.value if h.build_target else None,
                    }
                    for h in ps.history
                ],
                "draw_pile_count": ps.draw_pile_count,
                "discard_pile_count": ps.discard_pile_count,
            }
            # Per design/19 D3: only include draw_pile / discard_pile when not None.
            if ps.draw_pile is not None:
                result["draw_pile"] = [c.value for c in ps.draw_pile]
            if ps.discard_pile is not None:
                result["discard_pile"] = [c.value for c in ps.discard_pile]
            return result

        return {
            "round": gs.round,
            "cycle": gs.cycle,
            "round_in_cycle": gs.round_in_cycle,
            "players": {
                "1": _serialize_player(gs.players["1"]),
                "2": _serialize_player(gs.players["2"]),
            },
            "is_over": gs.is_over,
            "winner": gs.winner,
            "mode": gs.mode,
            "round_log": [RoundResult.to_dict(rr) for rr in gs.round_log],
        }
