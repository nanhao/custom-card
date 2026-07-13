"""Read-only game state snapshot for the custom card game v2 (design/34).

Provides GameStateSnapshot dataclass that replaces the dict returned by
GameCore.get_state(). Per design/19 D1, pure data container with typed
fields — no strings for cards or statuses.

Per architecture overview §2: game layer is pure logic — no UI, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.backend.game.player_state_snapshot import PlayerStateSnapshot
from src.backend.game.round_result import RoundResult


@dataclass
class GameStateSnapshot:
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
    players: dict[str, PlayerStateSnapshot]
    is_over: bool
    winner: str | None
    mode: str
    round_log: list[RoundResult]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize this GameStateSnapshot to a JSON-safe dict.

        Converts CardType → string via .value, PlayerStatus → dict via
        .to_dict(), PlayerHistoryItem → {"card": str, "build_target": str|None},
        RoundResult → dict via RoundResult.to_dict().

        Per design/19 D4: this is the serialization boundary — Converter and
        AppManager consume the typed GameStateSnapshot directly.

        Args:
            gs: The GameStateSnapshot value object to serialize.

        Returns:
            dict with nested players and round_log structures,
            matching the existing JSON API schema.
        """
        return {
            "round": self.round,
            "cycle": self.cycle,
            "round_in_cycle": self.round_in_cycle,
            "players": {
                "1": self.players["1"].to_dict(),
                "2": self.players["2"].to_dict(),
            },
            "is_over": self.is_over,
            "winner": self.winner,
            "mode": self.mode,
            "round_log": [RoundResult.to_dict(rr) for rr in self.round_log],
        }
