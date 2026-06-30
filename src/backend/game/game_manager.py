"""Game state machine for the custom card game.

Per architecture D9, this module is pure logic — no UI, no I/O.
Per architecture D11, the face-down selection protocol is handled
by external adapter layers (script/ or controller/).

The GameManager is the only stateful component in the game/ module.
It orchestrates the round flow: validate → resolve → damage → discard
→ record → check end → return result.
"""

from dataclasses import dataclass, field

from src.backend.game.card import CardType, get_initial_hand
from src.backend.game.player import Player
from src.backend.game.resolver import CardResolver, ResolutionResult


@dataclass
class RoundResult:
    """Value object returned by GameManager.play_round().

    Contains all data needed by adapter layers to display the round outcome.
    """

    round: int
    card_a: CardType
    card_b: CardType
    damage_a: int
    damage_b: int
    hp_a_after: int
    hp_b_after: int
    details: list[str]
    is_game_over: bool
    winner: str | None


class GameManager:
    """Manages the full game lifecycle.

    This is the only stateful component in the game/ module (architecture D9).
    Adapter layers (script/, controller/) call start_game() to initialize
    and play_round() to advance the game state.

    Attributes:
        player_a: Player A state (hp, hand, history).
        player_b: Player B state (hp, hand, history).
        round_number: Current round number (1–7).
        is_over: Whether the game has ended.
        winner: "A", "B", "draw", or None if game is ongoing.
        round_log: History of all completed rounds.
    """

    MAX_ROUNDS = 7

    def __init__(self) -> None:
        """Create an uninitialized GameManager. Call start_game() to begin."""
        self.player_a: Player | None = None
        self.player_b: Player | None = None
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
        self.player_a = Player(hp=3, hand=get_initial_hand())
        self.player_b = Player(hp=3, hand=get_initial_hand())
        self.round_number = 1
        self.is_over = False
        self.winner = None
        self.round_log = []

    def play_round(self, card_a: CardType, card_b: CardType) -> RoundResult:
        """Execute one round of the game.

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
            card_a: Card played by Player A this round.
            card_b: Card played by Player B this round.

        Returns:
            RoundResult with all round outcome data.

        Raises:
            ValueError: If game is over, card not in player's hand,
                        or players are not initialized.
        """
        # ---- Step 1: Validate ----
        self._validate(card_a, card_b)

        # ---- Step 2: Resolve ----
        resolution: ResolutionResult = CardResolver.resolve(
            card_a, card_b,
            self.player_a.history,
            self.player_b.history,
        )

        # ---- Step 3: Apply damage ----
        self.player_a.take_damage(resolution.damage_a)
        self.player_b.take_damage(resolution.damage_b)

        # ---- Step 4: Discard cards (after damage per D6) ----
        self.player_a.play_card(card_a)
        self.player_b.play_card(card_b)

        # ---- Step 5: Record history ----
        self.player_a.history.append(card_a)
        self.player_b.history.append(card_b)

        # ---- Step 6: Check game-end conditions (priority order) ----
        is_over, winner = self._check_game_end()

        # ---- Step 7: Build and return result ----
        result = RoundResult(
            round=self.round_number,
            card_a=card_a,
            card_b=card_b,
            damage_a=resolution.damage_a,
            damage_b=resolution.damage_b,
            hp_a_after=self.player_a.hp,
            hp_b_after=self.player_b.hp,
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
            dict with keys: round, players (A/B with hp/hand/history),
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
                "A": _player_dict(self.player_a),
                "B": _player_dict(self.player_b),
            },
            "is_over": self.is_over,
            "winner": self.winner,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, card_a: CardType, card_b: CardType) -> None:
        """Validate that play_round() can proceed.

        Raises ValueError if:
        - Game is already over (TC-X2)
        - Either card is not in the respective player's hand (TC-X3)
        - Players are not initialized
        """
        if self.is_over:
            raise ValueError("Cannot play round: the game is over.")
        if self.player_a is None or self.player_b is None:
            raise ValueError(
                "Cannot play round: game not started. Call start_game() first."
            )
        if not self.player_a.has_card(card_a):
            raise ValueError(
                f"Card {card_a.name} is not in Player A's hand."
            )
        if not self.player_b.has_card(card_b):
            raise ValueError(
                f"Card {card_b.name} is not in Player B's hand."
            )

    def _check_game_end(self) -> tuple[bool, str | None]:
        """Evaluate game-end conditions in priority order.

        Per game_flow.md §3, win condition priority:
        1. Both HP = 0 → draw
        2. A HP = 0, B HP > 0 → B wins
        3. B HP = 0, A HP > 0 → A wins
        4. Round 7 reached, A HP > B HP → A wins
        5. Round 7 reached, B HP > A HP → B wins
        6. Round 7 reached, A HP = B HP → draw

        Returns:
            (is_over, winner) tuple. winner is None if game continues.
        """
        hp_a = self.player_a.hp
        hp_b = self.player_b.hp

        # Priority 1: both dead
        if hp_a == 0 and hp_b == 0:
            self.is_over = True
            self.winner = "draw"
            return True, "draw"

        # Priority 2: A dead
        if hp_a == 0:
            self.is_over = True
            self.winner = "B"
            return True, "B"

        # Priority 3: B dead
        if hp_b == 0:
            self.is_over = True
            self.winner = "A"
            return True, "A"

        # Priority 4–6: round limit reached
        if self.round_number >= self.MAX_ROUNDS:
            self.is_over = True
            if hp_a > hp_b:
                self.winner = "A"
            elif hp_b > hp_a:
                self.winner = "B"
            else:
                self.winner = "draw"
            return True, self.winner

        # Game continues
        return False, None
