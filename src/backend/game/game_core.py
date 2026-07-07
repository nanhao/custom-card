"""Game state machine for the custom card game v2 (design/11).

Per design/11 D4, the v2 game introduces cycle mechanics: rounds are grouped
into cycles of 5, with cycle-end HP bleed and deck reshuffling. No maximum
round limit — game ends only when HP reaches 0.

Per architecture D9, this module is pure logic — no UI, no I/O.
"""

import random
from dataclasses import dataclass, field

from src.backend.game.card import CardType, get_card_def, get_initial_draw_pile
from src.backend.game.player import Player
from src.backend.game.resolver import CardResolver


@dataclass
class RoundResult:
    """Value object returned by GameCore.play_round().

    Attributes:
        round: Global round number.
        cycle: Current cycle number.
        round_in_cycle: Position within the cycle (1–5).
        cards_1, cards_2: Cards played this round.
        damage_1, damage_2: Damage dealt to each player.
        hp_1_after, hp_2_after: HP after resolution.
        energy_1_after, energy_2_after: Energy after resolution.
        statuses_1, statuses_2: Statuses after resolution.
        details: Human-readable resolution details.
        is_game_over: Whether the game ended this round.
        winner: "1", "2", "draw", or None.
    """

    round: int
    cycle: int
    round_in_cycle: int
    cards_1: list[CardType]
    cards_2: list[CardType]
    damage_1: int
    damage_2: int
    hp_1_after: int
    hp_2_after: int
    energy_1_after: int
    energy_2_after: int
    statuses_1: list[str]
    statuses_2: list[str]
    details: list[str]
    is_game_over: bool
    winner: str | None
    cycle_ended: bool = False


class GameCore:
    """Manages the full v2 game lifecycle with cycle mechanics.

    Key changes from v1 (design/11):
    - ROUNDS_PER_CYCLE = 5: games proceed in cycles with no max round limit.
    - cycle_number and round_in_cycle track position within a cycle.
    - _start_cycle() handles deck shuffling and card drawing.
    - Cycle-end HP bleed: both players lose 1 HP after round 5.
    - build_target parameter for BUILD card construction.
    """

    ROUNDS_PER_CYCLE = 5

    def __init__(self, mode: str = "pvp") -> None:
        """Create an uninitialized GameCore. Call start_game() to begin.

        Args:
            mode: Game mode — "pvp" (default) or "pve".
        """
        self.mode: str = mode
        self.player_1: Player | None = None
        self.player_2: Player | None = None
        self.round_number: int = 0
        self.cycle_number: int = 0
        self.round_in_cycle: int = 0
        self.is_over: bool = False
        self.winner: str | None = None
        self.round_log: list[RoundResult] = []
        # Status lifetime tracking: index-aligned with player.statuses
        self._status_lifetimes_1: list[int] = []
        self._status_lifetimes_2: list[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_game(self) -> None:
        """Initialize both players and start the first cycle.

        Sets HP=20, energy=0, initializes the draw pile with 7 cards,
        then triggers _start_cycle() to shuffle and deal hands.
        """
        initial_pile = get_initial_draw_pile()
        self.player_1 = Player(hp=20, draw_pile=list(initial_pile))
        self.player_2 = Player(hp=20, draw_pile=list(initial_pile))
        self.round_number = 1
        self.cycle_number = 1
        self.round_in_cycle = 1
        self.is_over = False
        self.winner = None
        self.round_log = []
        self._start_cycle()

    def play_round(
        self,
        cards_1: list[CardType],
        cards_2: list[CardType],
        build_target_1: CardType | None = None,
        build_target_2: CardType | None = None,
    ) -> RoundResult:
        """Execute one round of the game.

        Flow:
        1. Validate (game not over, cards in hand, BUILD energy check).
        2. Resolve card effects via CardResolver.
        3. Apply damage, energy, statuses.
        4. Process BUILD construction if applicable.
        5. Discard played cards.
        6. Record history.
        7. Check cycle end (round 5 → bleed + new cycle).
        8. Check game end (HP=0).
        9. Return RoundResult.

        Args:
            cards_1: Cards played by Player 1 (expects exactly 1).
            cards_2: Cards played by Player 2 (expects exactly 1).
            build_target_1: Card to construct if P1 plays BUILD.
            build_target_2: Card to construct if P2 plays BUILD.

        Returns:
            RoundResult with all round outcome data.

        Raises:
            ValueError: If game is over, cards not in hand, energy insufficient
                        for BUILD, or players are not initialized.
        """
        # Step 1: Validate
        self._validate(cards_1, cards_2, build_target_1, build_target_2)

        # Step 2: Compute attack streaks from history
        p1_streak = CardResolver.get_consecutive_attack(self.player_1.history)
        p2_streak = CardResolver.get_consecutive_attack(self.player_2.history)

        # Step 3: Resolve
        resolution = CardResolver.resolve(
            cards_1, cards_2,
            self.player_1.statuses,
            self.player_2.statuses,
            p1_streak,
            p2_streak,
        )

        # Step 4: Apply damage
        self.player_1.take_damage(resolution.damage_1)
        self.player_2.take_damage(resolution.damage_2)

        # Step 5: Apply energy and statuses
        self.player_1.add_energy(resolution.energy_delta_1)
        self.player_2.add_energy(resolution.energy_delta_2)
        # Replace statuses with the full set computed this round, lifetime=1 each
        self.player_1.statuses = list(resolution.statuses_granted_1)
        self.player_2.statuses = list(resolution.statuses_granted_2)
        self._status_lifetimes_1 = [1] * len(resolution.statuses_granted_1)
        self._status_lifetimes_2 = [1] * len(resolution.statuses_granted_2)

        # Step 6: Process BUILD construction
        self._process_build(cards_1[0], build_target_1, self.player_1)
        self._process_build(cards_2[0], build_target_2, self.player_2)

        # Step 7: Discard played cards from hands (to discard_pile)
        self.player_1.play_cards(cards_1)
        self.player_2.play_cards(cards_2)
        self.player_1.discard_pile.extend(cards_1)
        self.player_2.discard_pile.extend(cards_2)

        # Step 8: Record history
        self.player_1.history.extend(cards_1)
        self.player_2.history.extend(cards_2)

        # Step 9: Build initial result (pre-cycle-end check)
        result = RoundResult(
            round=self.round_number,
            cycle=self.cycle_number,
            round_in_cycle=self.round_in_cycle,
            cards_1=cards_1,
            cards_2=cards_2,
            damage_1=resolution.damage_1,
            damage_2=resolution.damage_2,
            hp_1_after=self.player_1.hp,
            hp_2_after=self.player_2.hp,
            energy_1_after=self.player_1.energy,
            energy_2_after=self.player_2.energy,
            statuses_1=list(self.player_1.statuses),
            statuses_2=list(self.player_2.statuses),
            details=resolution.details,
            is_game_over=False,
            winner=None,
            cycle_ended=False,
        )

        # Step 10: Check game end from damage
        is_over, winner = self._check_game_end()
        if is_over:
            result.is_game_over = True
            result.winner = winner
            self.round_log.append(result)
            return result

        # Step 11: Check cycle end — if round 5 complete, apply bleed + new cycle
        if self.round_in_cycle >= self.ROUNDS_PER_CYCLE:
            self.player_1.take_damage(1)
            self.player_2.take_damage(1)
            # Update result HP to reflect cycle bleed
            result.hp_1_after = self.player_1.hp
            result.hp_2_after = self.player_2.hp
            result.cycle_ended = True
            # Check game end again after cycle bleed
            is_over, winner = self._check_game_end()
            if is_over:
                result.is_game_over = True
                result.winner = winner
                self.round_log.append(result)
                return result
            # Start new cycle
            self.cycle_number += 1
            self.round_in_cycle = 0  # Will be incremented to 1 below
            self._start_cycle()

        # Step 12: Advance round
        self.round_number += 1
        self.round_in_cycle += 1

        # Expire statuses now that the round is fully complete
        self._tick_status_lifetimes()

        self.round_log.append(result)
        return result

    def get_state(self, player_perspective: str | None = None) -> dict:
        """Return a read-only snapshot of the current game state.

        Args:
            player_perspective: If "1" or "2", exposes that player's pile contents
                                and only opponent's pile counts. If None, both
                                players see only counts.

        Returns:
            dict with keys: round, cycle, round_in_cycle, players, is_over,
            winner, mode, round_log.
        """
        def _player_dict(p: Player | None, show_piles: bool) -> dict:
            if p is None:
                return {
                    "hp": 20, "energy": 0, "statuses": [],
                    "hand": [], "history": [],
                    "draw_pile_count": 0, "discard_pile_count": 0,
                }
            result = {
                "hp": p.hp,
                "energy": p.energy,
                "statuses": list(p.statuses),
                "hand": [c.value for c in p.hand],
                "history": [c.value for c in p.history],
            }
            if show_piles:
                result["draw_pile"] = [c.value for c in p.draw_pile]
                result["discard_pile"] = [c.value for c in p.discard_pile]
            else:
                result["draw_pile_count"] = len(p.draw_pile)
                result["discard_pile_count"] = len(p.discard_pile)
            return result

        show_p1 = (player_perspective == "1")
        show_p2 = (player_perspective == "2")

        return {
            "round": self.round_number,
            "cycle": self.cycle_number,
            "round_in_cycle": self.round_in_cycle,
            "players": {
                "1": _player_dict(self.player_1, show_p1),
                "2": _player_dict(self.player_2, show_p2),
            },
            "is_over": self.is_over,
            "winner": self.winner,
            "mode": self.mode,
            "round_log": [
                {
                    "round": rr.round,
                    "cycle": rr.cycle,
                    "round_in_cycle": rr.round_in_cycle,
                    "cards_1": [c.value for c in rr.cards_1],
                    "cards_2": [c.value for c in rr.cards_2],
                    "damage_1": rr.damage_1,
                    "damage_2": rr.damage_2,
                    "hp_1_after": rr.hp_1_after,
                    "hp_2_after": rr.hp_2_after,
                    "energy_1_after": rr.energy_1_after,
                    "energy_2_after": rr.energy_2_after,
                    "statuses_1": rr.statuses_1,
                    "statuses_2": rr.statuses_2,
                    "details": rr.details,
                    "is_game_over": rr.is_game_over,
                    "winner": rr.winner,
                    "cycle_ended": rr.cycle_ended,
                }
                for rr in self.round_log
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tick_status_lifetimes(self) -> None:
        """Decrement all status lifetimes by 1, remove expired ones.

        Called at the start of each round. Each status has a remaining
        lifetime in rounds; when it reaches 0 the status is removed.
        """
        for player, lifetimes in (
            (self.player_1, self._status_lifetimes_1),
            (self.player_2, self._status_lifetimes_2),
        ):
            if player is None:
                continue
            new_statuses = []
            new_lifetimes = []
            for s, life in zip(player.statuses, lifetimes):
                remaining = life - 1
                if remaining > 0:
                    new_statuses.append(s)
                    new_lifetimes.append(remaining)
            player.statuses = new_statuses
            if player is self.player_1:
                self._status_lifetimes_1 = new_lifetimes
            else:
                self._status_lifetimes_2 = new_lifetimes

    def _start_cycle(self) -> None:
        """Start a new cycle: reclaim hands → shuffle discard into draw → draw 7 cards.

        Per design/11 D4:
        1. Move all cards from hands to discard_pile.
        2. Move all cards from discard_pile to draw_pile.
        3. Shuffle the draw_pile.
        4. Draw 7 cards from draw_pile into hand.
        """
        for player in (self.player_1, self.player_2):
            if player is None:
                continue
            # Move hand cards to discard
            player.discard_pile.extend(player.hand)
            player.hand = []
            # Move discard to draw pile
            player.draw_pile.extend(player.discard_pile)
            player.discard_pile = []
            # Shuffle and draw 7
            random.shuffle(player.draw_pile)
            draw_count = min(7, len(player.draw_pile))
            player.hand = player.draw_pile[:draw_count]
            player.draw_pile = player.draw_pile[draw_count:]

    def _validate(
        self,
        cards_1: list[CardType],
        cards_2: list[CardType],
        build_target_1: CardType | None,
        build_target_2: CardType | None,
    ) -> None:
        """Validate that play_round() can proceed.

        Raises ValueError if:
        - Game is over.
        - Players are not initialized.
        - Any card is not in the respective player's hand.
        - BUILD played with energy=0.
        - BUILD played without build_target.
        - BUILD target costs more energy than available.
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

        # BUILD validation for P1
        if CardType.BUILD in cards_1:
            if self.player_1.energy == 0:
                raise ValueError(
                    "Cannot play BUILD: energy is 0."
                )
            if build_target_1 is None:
                raise ValueError(
                    "Cannot play BUILD: build_target_1 is required."
                )
            build_cost = get_card_def(build_target_1).build_cost
            if build_cost > self.player_1.energy:
                raise ValueError(
                    f"Cannot build {build_target_1.value}: "
                    f"cost {build_cost} > energy {self.player_1.energy}."
                )

        # BUILD validation for P2
        if CardType.BUILD in cards_2:
            if self.player_2.energy == 0:
                raise ValueError(
                    "Cannot play BUILD: energy is 0."
                )
            if build_target_2 is None:
                raise ValueError(
                    "Cannot play BUILD: build_target_2 is required."
                )
            build_cost = get_card_def(build_target_2).build_cost
            if build_cost > self.player_2.energy:
                raise ValueError(
                    f"Cannot build {build_target_2.value}: "
                    f"cost {build_cost} > energy {self.player_2.energy}."
                )

    def _process_build(
        self,
        card: CardType,
        build_target: CardType | None,
        player: Player,
    ) -> None:
        """Process BUILD card effect: deduct energy and add built card to discard.

        Args:
            card: The card played (should be BUILD for this to take effect).
            build_target: The card type to construct.
            player: The player who played the build card.
        """
        if card != CardType.BUILD or build_target is None:
            return
        cost = get_card_def(build_target).build_cost
        player.energy = max(0, player.energy - cost)
        player.discard_pile.append(build_target)

    def _check_game_end(self) -> tuple[bool, str | None]:
        """Evaluate game-end conditions: HP=0 only (no round limit in v2).

        Priority order:
        1. Both HP = 0 → draw
        2. P1 HP = 0 → P2 wins
        3. P2 HP = 0 → P1 wins

        Returns:
            (is_over, winner) tuple. winner is None if game continues.
        """
        hp_1 = self.player_1.hp
        hp_2 = self.player_2.hp

        if hp_1 == 0 and hp_2 == 0:
            self.is_over = True
            self.winner = "draw"
            return True, "draw"
        if hp_1 == 0:
            self.is_over = True
            self.winner = "2"
            return True, "2"
        if hp_2 == 0:
            self.is_over = True
            self.winner = "1"
            return True, "1"

        return False, None
