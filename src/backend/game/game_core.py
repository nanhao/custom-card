"""Game state machine for the custom card game.

Per architecture D9, this module is pure logic — no UI, no I/O.
Per architecture D11, the face-down selection protocol is handled
by external adapter layers (script/ or controller/).

The GameCore is the only stateful component in the game/ module.
It orchestrates the round flow: validate → resolve → damage → discard
→ record → check end → return result.
"""

from dataclasses import dataclass, field

from src.backend.game.card import CardType, get_initial_hand
from src.backend.game.player import Player
from src.backend.game.resolver import CardResolver, ResolutionResult


@dataclass
class RoundResult:
    """Value object returned by GameCore.play_round().

    Contains all data needed by adapter layers to display the round outcome.
    Uses list[CardType] for cards_1/cards_2 to support future multi-card
    gameplay (architecture D12).
    """

    round: int
    cards_1: list[CardType]
    cards_2: list[CardType]
    damage_1: int
    damage_2: int
    hp_1_after: int
    hp_2_after: int
    details: list[str]
    is_game_over: bool
    winner: str | None


class GameCore:
    """Manages the full game lifecycle.

    This is the only stateful component in the game/ module (architecture D9).
    Adapter layers (script/, controller/) call start_game() to initialize
    and play_round() to advance the game state.

    Attributes:
        player_1: Player 1 state (hp, hand, history).
        player_2: Player 2 state (hp, hand, history).
        round_number: Current round number (1–7).
        is_over: Whether the game has ended.
        winner: "1", "2", "draw", or None if game is ongoing.
        round_log: History of all completed rounds.
    """

    MAX_ROUNDS = 7

    def __init__(self, mode: str = "pvp") -> None:
        """Create an uninitialized GameCore. Call start_game() to begin.

        Args:
            mode: Game mode — "pvp" (default) or "pve".
                  Set at construction, immutable thereafter (per D2).
        """
        self.mode: str = mode
        self.player_1: Player | None = None
        self.player_2: Player | None = None
        self.round_number: int = 0
        self.is_over: bool = False
        self.winner: str | None = None
        self.round_log: list[RoundResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_game(self) -> None:
        """Initialize both players and reset all game state.

        Sets HP to 3, deals the standard 7-card hand to each player,
        sets round to 1, and clears the round log. Can be called at
        any time to restart (doubles as reset — TC-X6).
        """
        self.player_1 = Player(hp=3, hand=get_initial_hand())
        self.player_2 = Player(hp=3, hand=get_initial_hand())
        self.round_number = 1
        self.is_over = False
        self.winner = None
        self.round_log = []

    def play_round(self, cards_1: list[CardType], cards_2: list[CardType]) -> RoundResult:
        """Execute one round of the game.

        Accepts lists for future multi-card support (architecture D12).
        Current version expects exactly 1 card per list.

        Follows the round execution flow defined in game_flow.md §2:
        1. Validate inputs (game not over, cards in hands)
        2. Resolve card effects via CardResolver
        3. Apply damage to both players
        4. Discard played cards from hands
        5. Record cards in player histories
        6. Check game-end conditions
        7. Return RoundResult

        Per architecture D6: cards are discarded AFTER damage is applied,
        so resolution never depends on current hand state.

        Args:
            cards_1: Cards played by Player 1 this round.
            cards_2: Cards played by Player 2 this round.

        Returns:
            RoundResult with all round outcome data.

        Raises:
            ValueError: If game is over, any card not in player's hand,
                        or players are not initialized.
        """
        # ---- Step 1: Validate ----
        self._validate(cards_1, cards_2)

        # ---- Step 2: Resolve ----
        resolution: ResolutionResult = CardResolver.resolve(
            cards_1, cards_2,
            self.player_1.history,
            self.player_2.history,
        )

        # ---- Step 3: Apply damage ----
        self.player_1.take_damage(resolution.damage_1)
        self.player_2.take_damage(resolution.damage_2)

        # ---- Step 4: Discard cards (after damage per D6) ----
        self.player_1.play_cards(cards_1)
        self.player_2.play_cards(cards_2)

        # ---- Step 5: Record history ----
        self.player_1.history.extend(cards_1)
        self.player_2.history.extend(cards_2)

        # ---- Step 6: Check game-end conditions (priority order) ----
        is_over, winner = self._check_game_end()

        # ---- Step 7: Build and return result ----
        result = RoundResult(
            round=self.round_number,
            cards_1=cards_1,
            cards_2=cards_2,
            damage_1=resolution.damage_1,
            damage_2=resolution.damage_2,
            hp_1_after=self.player_1.hp,
            hp_2_after=self.player_2.hp,
            details=resolution.details,
            is_game_over=is_over,
            winner=winner,
        )
        self.round_log.append(result)

        # Advance round if game continues
        if not is_over:
            self.round_number += 1

        return result

    def get_state(self) -> dict:
        """Return a read-only snapshot of the current game state.

        Used by adapter layers (script/, controller/) to render UI
        or serialize state for API responses.

        Returns:
            dict with keys: round, players (1/2 with hp/hand/history),
            is_over, winner.
        """
        def _player_dict(p: Player | None) -> dict:
            if p is None:
                return {"hp": 3, "hand": [], "history": []}
            return {
                "hp": p.hp,
                "hand": [c.value for c in p.hand],
                "history": [c.value for c in p.history],
            }

        return {
            "round": self.round_number,
            "players": {
                "1": _player_dict(self.player_1),
                "2": _player_dict(self.player_2),
            },
            "is_over": self.is_over,
            "winner": self.winner,
            "mode": self.mode,
            "round_log": [
                {
                    "round": rr.round,
                    "cards_1": [c.value for c in rr.cards_1],
                    "cards_2": [c.value for c in rr.cards_2],
                    "damage_1": rr.damage_1,
                    "damage_2": rr.damage_2,
                    "hp_1_after": rr.hp_1_after,
                    "hp_2_after": rr.hp_2_after,
                    "details": rr.details,
                    "is_game_over": rr.is_game_over,
                    "winner": rr.winner,
                }
                for rr in self.round_log
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, cards_1: list[CardType], cards_2: list[CardType]) -> None:
        """Validate that play_round() can proceed.

        Raises ValueError if:
        - Game is already over (TC-X2)
        - Any card is not in the respective player's hand (TC-X3)
        - Players are not initialized
        """
        if self.is_over:
            raise ValueError("Cannot play round: the game is over.")
        if self.player_1 is None or self.player_2 is None:
            raise ValueError(
                "Cannot play round: game not started. Call start_game() first."
            )
        if not self.player_1.has_cards(cards_1):
            raise ValueError(
                f"Not all cards in {[c.name for c in cards_1]} are in Player 1's hand."
            )
        if not self.player_2.has_cards(cards_2):
            raise ValueError(
                f"Not all cards in {[c.name for c in cards_2]} are in Player 2's hand."
            )

    def _check_game_end(self) -> tuple[bool, str | None]:
        """Evaluate game-end conditions in priority order.

        Per game_flow.md §3, win condition priority:
        1. Both HP = 0 → draw
        2. 1 HP = 0, 2 HP > 0 → 2 wins
        3. 2 HP = 0, 1 HP > 0 → 1 wins
        4. Round 7 reached, 1 HP > 2 HP → 1 wins
        5. Round 7 reached, 2 HP > 1 HP → 2 wins
        6. Round 7 reached, 1 HP = 2 HP → draw

        Returns:
            (is_over, winner) tuple. winner is None if game continues.
        """
        hp_1 = self.player_1.hp
        hp_2 = self.player_2.hp

        # Priority 1: both dead
        if hp_1 == 0 and hp_2 == 0:
            self.is_over = True
            self.winner = "draw"
            return True, "draw"

        # Priority 2: 1 dead
        if hp_1 == 0:
            self.is_over = True
            self.winner = "2"
            return True, "2"

        # Priority 3: 2 dead
        if hp_2 == 0:
            self.is_over = True
            self.winner = "1"
            return True, "1"

        # Priority 4–6: round limit reached
        if self.round_number >= self.MAX_ROUNDS:
            self.is_over = True
            if hp_1 > hp_2:
                self.winner = "1"
            elif hp_2 > hp_1:
                self.winner = "2"
            else:
                self.winner = "draw"
            return True, self.winner

        # Game continues
        return False, None
