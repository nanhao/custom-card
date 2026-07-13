"""Game state machine for the custom card game v2 (design/11, design/26, design/35).

Per design/11 D4, the v2 game introduces cycle mechanics: rounds are grouped
into cycles of 3 (design/26 D1, was 5), with cycle-end HP bleed and deck
reshuffling. No maximum round limit — game ends only when HP reaches 0.

Per design/26 D2: CARDS_PER_CYCLE = 5 (was hardcoded 7), initial draw pile
stays at 7 cards (D3), so 2 cards remain in draw pile after first deal.

Per design/35: classic DBG draw — discard pile only shuffled into draw pile
when draw pile is empty during draw_cards(), instead of at every cycle start.

Per architecture D9, this module is pure logic — no UI, no I/O.
"""

import random

from src.backend.game.card_type import CardType
from src.backend.game.card_def import get_card_def, get_initial_draw_pile
from src.backend.game.constants import ROUNDS_PER_CYCLE, CARDS_PER_CYCLE, INITIAL_HP, INITIAL_ENERGY
from src.backend.game.game_state_snapshot import GameStateSnapshot
from src.backend.game.player_state_snapshot import PlayerStateSnapshot
from src.backend.game.player import Player
from src.backend.game.player_history_item import PlayerHistoryItem
from src.backend.game.player_state import StatusType, PlayerStatus
from src.backend.game.resolver import CardResolver
from src.backend.game.resolver.base import ResolveContext, Resolver
from src.backend.game.round_result import (
    BeforeResolve,
    AfterResolve,
    EndActions,
    AfterEndActions,
    RoundResult,
)


class GameCore:
    """Manages the full v2 game lifecycle with cycle mechanics.

    Key changes from v1 (design/11, design/26):
    - ROUNDS_PER_CYCLE = 3 (design/26 D1, was 5): 3 rounds per cycle.
    - CARDS_PER_CYCLE = 5 (design/26 D2): draw 5 cards per cycle start.
    - cycle_number and round_in_cycle track position within a cycle.
    - _start_cycle() handles deck shuffling and card drawing.
    - Cycle-end HP bleed: both players lose 1 HP after round 3 (was 5).
    - build_target parameter for BUILD card construction.
    - BUILD settlement handled by BuildResolver (design/25).
    """

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_game(self) -> None:
        """Initialize both players and start the first cycle.

        Per design/38: Sets HP=30, energy=2, initializes the draw pile
        with 7 cards, then triggers _start_cycle() to shuffle and deal
        5 cards per hand.
        Per design/26 D3: 2 cards remain in draw pile.
        """
        initial_pile = get_initial_draw_pile()
        self.player_1 = Player(hp=INITIAL_HP, energy=INITIAL_ENERGY, draw_pile=list(initial_pile))
        self.player_2 = Player(hp=INITIAL_HP, energy=INITIAL_ENERGY, draw_pile=list(initial_pile))
        # Shuffle draw piles so cards are dealt randomly (DBG standard).
        random.shuffle(self.player_1.draw_pile)
        random.shuffle(self.player_2.draw_pile)
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
        dismantle_target_1: CardType | None = None,
        dismantle_target_2: CardType | None = None,
    ) -> RoundResult:
        """Execute one round of the game.

        Flow:
        1. Validate (game not over, cards in hand, BUILD dual-mode check).
        2. Build resolver tasks with build_target/dismantle_target via ResolveContext.
        3. Resolve card effects via CardResolver.
        4. Discard played cards.
        5. Record history.
        6. Check cycle end (round 3 → bleed + new cycle).
        7. Check game end (HP=0).
        8. Return RoundResult.

        Args:
            cards_1: Cards played by Player 1 (empty list = pass).
            cards_2: Cards played by Player 2 (empty list = pass).
            build_target_1: Card to construct if P1 plays BUILD (build mode).
            build_target_2: Card to construct if P2 plays BUILD (build mode).
            dismantle_target_1: Card to remove if P1 plays BUILD (dismantle mode).
            dismantle_target_2: Card to remove if P2 plays BUILD (dismantle mode).

        Returns:
            RoundResult with all round outcome data.

        Raises:
            ValueError: If game is over, cards not in hand, energy insufficient
                        for BUILD, or players are not initialized.
        """
        # Step 1: Validate
        self._validate(
            cards_1, cards_2,
            build_target_1, build_target_2,
            dismantle_target_1, dismantle_target_2,
        )

        # Step 1.5: Capture before-resolve snapshot.
        # Record HP, energy, and statuses BEFORE any card effects are applied.
        before = BeforeResolve(
            statuses_1=[PlayerStatus(ps.status_type, ps.lifetime) for ps in self.player_1.statuses],
            statuses_2=[PlayerStatus(ps.status_type, ps.lifetime) for ps in self.player_2.statuses],
            energy_1=self.player_1.energy,
            energy_2=self.player_2.energy,
            hp_1=self.player_1.hp,
            hp_2=self.player_2.hp,
        )

        # Step 2: Build resolver tasks from CardDef.resolvers (design/22).
        # GameCore owns the mapping lookup; CardResolver only orchestrates.
        # Per design/25: build_target+build_target_cost passed via ResolveContext.
        # Per design/28: empty cards list = pass — no resolver tasks for that player.
        tasks: list[tuple[Resolver, ResolveContext]] = []

        if cards_1:
            for card_type in cards_1:
                build_target = build_target_1 if card_type == CardType.BUILD else None
                build_cost = get_card_def(build_target).build_cost if build_target is not None else 0
                ctx = ResolveContext(
                    card_owner=self.player_1,
                    opponent=self.player_2,
                    card_played_by_owner=card_type,
                    card_played_by_opponent=cards_2[0] if cards_2 else None,
                    build_target=build_target,
                    build_target_cost=build_cost,
                    dismantle_target=dismantle_target_1,
                )
                for resolver_cls in get_card_def(card_type).resolvers:
                    tasks.append((resolver_cls(), ctx))

        if cards_2:
            for card_type in cards_2:
                build_target = build_target_2 if card_type == CardType.BUILD else None
                build_cost = get_card_def(build_target).build_cost if build_target is not None else 0
                ctx = ResolveContext(
                    card_owner=self.player_2,
                    opponent=self.player_1,
                    card_played_by_owner=card_type,
                    card_played_by_opponent=cards_1[0] if cards_1 else None,
                    build_target=build_target,
                    build_target_cost=build_cost,
                    dismantle_target=dismantle_target_2,
                )
                for resolver_cls in get_card_def(card_type).resolvers:
                    tasks.append((resolver_cls(), ctx))

        resolution = CardResolver.resolve(tasks)

        # Step 3 removed: BUILD construction now handled by BuildResolver (design/25).

        # Step 4: Discard played cards from hands (to discard_pile)
        # Per design/28: empty cards list = pass — no discard.
        if cards_1:
            self.player_1.play_cards(cards_1)
            self.player_1.discard_pile.extend(cards_1)
        if cards_2:
            self.player_2.play_cards(cards_2)
            self.player_2.discard_pile.extend(cards_2)

        # Step 5: Record history as PlayerHistoryItem (design/16 D3)
        # Per design/28: pass records card=None, build_target=None.
        # Per design/37: dismantle_target recorded for BUILD dismantle mode.
        self.player_1.history.append(
            PlayerHistoryItem(
                cards_1[0] if cards_1 else None,
                build_target_1 if cards_1 and CardType.BUILD in cards_1 else None,
                dismantle_target_1 if cards_1 and CardType.BUILD in cards_1 else None,
            )
        )
        self.player_2.history.append(
            PlayerHistoryItem(
                cards_2[0] if cards_2 else None,
                build_target_2 if cards_2 and CardType.BUILD in cards_2 else None,
                dismantle_target_2 if cards_2 and CardType.BUILD in cards_2 else None,
            )
        )

        # Step 6: after_resolve — snapshot after card resolution, before round-end.
        after_resolve = AfterResolve(
            statuses_1=[PlayerStatus(ps.status_type, ps.lifetime) for ps in self.player_1.statuses],
            statuses_2=[PlayerStatus(ps.status_type, ps.lifetime) for ps in self.player_2.statuses],
            energy_1=self.player_1.energy,
            energy_2=self.player_2.energy,
            hp_1=self.player_1.hp,
            hp_2=self.player_2.hp,
            details=resolution.details,
        )

        # Step 7: Check game end from damage
        is_over, winner = self._check_game_end()
        if is_over:
            # Game ended during card resolution — after_end_actions copies after_resolve state.
            after_resolve.is_game_over = True
            after_resolve.winner = winner
            after_end_actions = AfterEndActions(
                is_game_over=True,
                winner=winner,
                round=self.round_number,
                cycle=self.cycle_number,
                round_in_cycle=self.round_in_cycle,
                hp_1=after_resolve.hp_1,
                hp_2=after_resolve.hp_2,
                statuses_1=list(after_resolve.statuses_1),
                statuses_2=list(after_resolve.statuses_2),
            )
            result = RoundResult(
                round=self.round_number,
                cycle=self.cycle_number,
                round_in_cycle=self.round_in_cycle,
                cards_1=cards_1,
                cards_2=cards_2,
                before=before,
                after_resolve=after_resolve,
                end_actions=EndActions(),
                after_end_actions=after_end_actions,
            )
            self.round_log.append(result)
            return result

        # Step 8: Phase 2 — Round-end actions
        # a) Status lifetime tick (every round)
        self._tick_status_lifetimes()

        # b) Cycle-end processing (round 3)
        end_actions = EndActions()
        if self.round_in_cycle >= ROUNDS_PER_CYCLE:
            end_actions.cycle_ended = True
            end_actions.bleed_damage_1 = 1
            end_actions.bleed_damage_2 = 1
            self.player_1.take_damage(1)
            self.player_2.take_damage(1)
            bleed_over, bleed_winner = self._check_game_end()
            self.cycle_number += 1
            self.round_in_cycle = 0
            self._start_cycle()
        else:
            bleed_over = False
            bleed_winner = None

        # Step 9: Advance round
        self.round_number += 1
        self.round_in_cycle += 1

        # Step 10: after_end_actions — final state after all processing.
        after_end_actions = AfterEndActions(
            is_game_over=bleed_over,
            winner=bleed_winner,
            round=self.round_number,
            cycle=self.cycle_number,
            round_in_cycle=self.round_in_cycle,
            hand_cards_1=list(self.player_1.hand) if self.player_1 else [],
            hand_cards_2=list(self.player_2.hand) if self.player_2 else [],
            statuses_1=[PlayerStatus(ps.status_type, ps.lifetime) for ps in self.player_1.statuses],
            statuses_2=[PlayerStatus(ps.status_type, ps.lifetime) for ps in self.player_2.statuses],
            energy_1=self.player_1.energy,
            energy_2=self.player_2.energy,
            hp_1=self.player_1.hp,
            hp_2=self.player_2.hp,
        )

        result = RoundResult(
            round=self.round_number - 1,  # pre-advancement round number
            cycle=self.cycle_number if not end_actions.cycle_ended else self.cycle_number - 1,
            round_in_cycle=self.round_in_cycle - 1 if not end_actions.cycle_ended else ROUNDS_PER_CYCLE,
            cards_1=cards_1,
            cards_2=cards_2,
            before=before,
            after_resolve=after_resolve,
            end_actions=end_actions,
            after_end_actions=after_end_actions,
        )

        self.round_log.append(result)
        return result

    def get_state(self, player_perspective: str | None = None) -> GameStateSnapshot:
        """Return a read-only typed snapshot of the current game state.

        Per design/19 D2: returns a GameStateSnapshot dataclass instead of a dict.
        PlayerStateSnapshot fields use typed objects (CardType, PlayerStatus) — no
        string or dict representations.

        Per design/34 D3: both players' draw_pile and discard_pile are always
        fully visible regardless of player_perspective. The parameter is kept
        for backward compatibility but does not affect behavior.

        Args:
            player_perspective: Deprecated in design/34. Kept for compatibility.
                                Always returns full piles for both players.

        Returns:
            GameStateSnapshot dataclass with typed fields.
        """

        def _build_player_state(p: Player | None) -> PlayerStateSnapshot:
            """Build a PlayerStateSnapshot from a Player instance."""
            if p is None:
                return PlayerStateSnapshot(
                    hp=INITIAL_HP, energy=INITIAL_ENERGY, statuses=[], hand=[], history=[],
                )
            # Deep-copy statuses and hand to prevent mutation of the snapshot.
            statuses = [
                PlayerStatus(ps.status_type, ps.lifetime) for ps in p.statuses
            ]
            hand = list(p.hand)
            history = list(p.history)
            # Per design/34: piles always fully visible to both players.
            return PlayerStateSnapshot(
                hp=p.hp, energy=p.energy,
                statuses=statuses, hand=hand, history=history,
                draw_pile=list(p.draw_pile),
                discard_pile=list(p.discard_pile),
                draw_pile_count=len(p.draw_pile),
                discard_pile_count=len(p.discard_pile),
            )

        return GameStateSnapshot(
            round=self.round_number,
            cycle=self.cycle_number,
            round_in_cycle=self.round_in_cycle,
            players={
                "1": _build_player_state(self.player_1),
                "2": _build_player_state(self.player_2),
            },
            is_over=self.is_over,
            winner=self.winner,
            mode=self.mode,
            round_log=list(self.round_log),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tick_status_lifetimes(self) -> None:
        """Decrement all status lifetimes by 1, remove expired ones.

        Per design/15 D4: operates on PlayerStatus.lifetime.
        Creates new PlayerStatus objects instead of mutating in place
        to prevent side effects on any external references.
        """
        for player in (self.player_1, self.player_2):
            if player is None:
                continue
            player.statuses = [
                PlayerStatus(ps.status_type, ps.lifetime - 1)
                for ps in player.statuses
                if ps.lifetime > 1
            ]

    def _start_cycle(self) -> None:
        """Start a new cycle: reclaim hands → draw CARDS_PER_CYCLE cards.

        Per design/35 (classic DBG draw):
        1. Move all cards from hands to discard_pile.
        2. Use Player.draw_cards(CARDS_PER_CYCLE) to draw cards with
           smart reshuffle (discard→draw only when draw pile empty).
        """
        for player in (self.player_1, self.player_2):
            if player is None:
                continue
            # Move hand cards to discard
            player.discard_pile.extend(player.hand)
            player.hand = []
            # Smart draw: shuffles discard into draw only when needed
            player.draw_cards(CARDS_PER_CYCLE)

    def _validate(
        self,
        cards_1: list[CardType],
        cards_2: list[CardType],
        build_target_1: CardType | None,
        build_target_2: CardType | None,
        dismantle_target_1: CardType | None = None,
        dismantle_target_2: CardType | None = None,
    ) -> None:
        """Validate that play_round() can proceed.

        Per design/28: empty cards list = pass — skip all validation for that player.
        Per design/37: BUILD dual-mode — build_target and dismantle_target are
        mutually exclusive. Build mode requires energy; dismantle mode does not.

        Raises ValueError if:
        - Game is over.
        - Players are not initialized.
        - Any card is not in the respective player's hand.
        - BUILD played with both or neither of build_target/dismantle_target.
        - BUILD build mode: energy=0 or insufficient energy for target.
        - BUILD dismantle mode: dismantle_target not in discard pile.
        - Non-BUILD card has a build_target or dismantle_target set.
        """
        if self.is_over:
            raise ValueError("Cannot play round: the game is over.")
        if self.player_1 is None or self.player_2 is None:
            raise ValueError(
                "Cannot play round: game not started. Call start_game() first."
            )

        # Per design/28: empty cards = pass, skip hand/BUILD validation
        if cards_1:
            if not self.player_1.has_cards(cards_1):
                raise ValueError(
                    f"Not all cards in {[c.name for c in cards_1]} are in Player 1's hand."
                )
            # BUILD validation for P1 — dual-mode (design/37)
            if CardType.BUILD in cards_1:
                self._validate_build_dual_mode(
                    self.player_1, build_target_1, dismantle_target_1,
                )
            else:
                # Non-BUILD cards must not have build/dismantle targets
                if build_target_1 is not None or dismantle_target_1 is not None:
                    raise ValueError(
                        "build_target_1/dismantle_target_1 must be None "
                        "when not playing BUILD."
                    )

        # Per design/28: empty cards = pass, skip hand/BUILD validation
        if cards_2:
            if not self.player_2.has_cards(cards_2):
                raise ValueError(
                    f"Not all cards in {[c.name for c in cards_2]} are in Player 2's hand."
                )
            # BUILD validation for P2 — dual-mode (design/37)
            if CardType.BUILD in cards_2:
                self._validate_build_dual_mode(
                    self.player_2, build_target_2, dismantle_target_2,
                )
            else:
                if build_target_2 is not None or dismantle_target_2 is not None:
                    raise ValueError(
                        "build_target_2/dismantle_target_2 must be None "
                        "when not playing BUILD."
                    )

    def _validate_build_dual_mode(
        self,
        player: Player,
        build_target: CardType | None,
        dismantle_target: CardType | None,
    ) -> None:
        """Validate BUILD card dual-mode: build and dismantle are mutually exclusive.

        Per design/37: when playing BUILD, exactly one of build_target or
        dismantle_target must be non-None.

        Args:
            player: The player playing BUILD.
            build_target: Card to build (build mode), or None.
            dismantle_target: Card to dismantle (dismantle mode), or None.

        Raises:
            ValueError: If both or neither targets are set, if build mode has
                        insufficient energy, or dismantle target not in discard.
        """
        # Must have exactly one target (mutually exclusive)
        if build_target is None and dismantle_target is None:
            raise ValueError(
                "Cannot play BUILD: must specify either build_target or "
                "dismantle_target."
            )
        if build_target is not None and dismantle_target is not None:
            raise ValueError(
                "Cannot play BUILD: build_target and dismantle_target are "
                "mutually exclusive."
            )

        # Build mode validation
        if build_target is not None:
            if player.energy == 0:
                raise ValueError(
                    "Cannot play BUILD (build mode): energy is 0."
                )
            build_cost = get_card_def(build_target).build_cost
            if build_cost > player.energy:
                raise ValueError(
                    f"Cannot build {build_target.value}: "
                    f"cost {build_cost} > energy {player.energy}."
                )

        # Dismantle mode validation
        if dismantle_target is not None:
            if dismantle_target not in player.discard_pile:
                raise ValueError(
                    f"Cannot dismantle {dismantle_target.value}: "
                    f"not found in discard pile."
                )

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

        if hp_1 <= 0 and hp_2 <= 0:
            self.is_over = True
            self.winner = "draw"
            return True, "draw"
        if hp_1 <= 0:
            self.is_over = True
            self.winner = "2"
            return True, "2"
        if hp_2 <= 0:
            self.is_over = True
            self.winner = "1"
            return True, "1"

        return False, None
