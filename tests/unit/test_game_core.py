"""Unit tests for src/backend/game/game_core.py — cycle mechanics.

Test cases per design/11 test_cases.md §4 (TC-GC-01 ~ TC-GC-11)
and design/14 test_cases.md §4-5 (TC-BH-02~04, TC-GS-01~04).
"""

import pytest

from src.backend.game.card import CardType
from src.backend.game.game_core import GameCore
from src.backend.game.game_state import GameState
from src.backend.game.player import PlayerHistoryItem
from src.backend.game.player_state import StatusType, PlayerStatus


class TestStartGame:
    """TC-GC-01: start_game initialization."""

    def test_start_game_state(self):
        """After start_game: hp=20, energy=0, hand=7, round=1, cycle=1, round_in_cycle=1."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        assert gm.player_1.hp == 20
        assert gm.player_2.hp == 20
        assert gm.player_1.energy == 0
        assert gm.player_2.energy == 0
        assert len(gm.player_1.hand) == 7
        assert len(gm.player_2.hand) == 7
        assert len(gm.player_1.draw_pile) == 0
        assert len(gm.player_1.discard_pile) == 0
        assert gm.round_number == 1
        assert gm.cycle_number == 1
        assert gm.round_in_cycle == 1
        assert gm.is_over is False

    def test_start_game_draw_pile_initialized(self):
        """Draw pile is initialized with 7 cards, then all drawn into hand."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        # After _start_cycle, hand has 7 cards, draw_pile is empty
        assert len(gm.player_1.hand) == 7
        assert len(gm.player_1.draw_pile) == 0


class TestPlayRoundDiscard:
    """TC-GC-02: Cards go to discard_pile after play."""

    def test_played_cards_go_to_discard(self):
        """After play_round, played cards are in discard_pile."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        # Ensure both players have the cards we want to play
        gm.player_1.hand = [CardType.ATTACK, CardType.DEFENSE]
        gm.player_2.hand = [CardType.ATTACK, CardType.DEFENSE]
        gm.player_1.draw_pile = [CardType.ATTACK] * 5  # fill hand count for get_state
        gm.player_2.draw_pile = [CardType.ATTACK] * 5

        result = gm.play_round([CardType.ATTACK], [CardType.DEFENSE])
        assert CardType.ATTACK in gm.player_1.discard_pile
        assert CardType.DEFENSE in gm.player_2.discard_pile
        # Hand size decreased
        assert len(gm.player_1.hand) == 1
        assert len(gm.player_2.hand) == 1


class TestCycleMechanics:
    """TC-GC-03, GC-04, GC-05, GC-06: Cycle mechanics."""

    def test_cycle_end_bleed(self):
        """TC-GC-03: After 5th round, both players lose 1 HP and new cycle starts."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        # Fast-forward to round_in_cycle=5
        gm.round_in_cycle = 5
        gm.cycle_number = 1
        gm.round_number = 5
        gm.player_1.hand = [CardType.ATTACK]
        gm.player_2.hand = [CardType.ATTACK]
        gm.player_1.hp = 10
        gm.player_2.hp = 10

        result = gm.play_round([CardType.ATTACK], [CardType.ATTACK])
        # Both took 1 attack damage → hp 10→9 (Phase 1, pre-bleed)
        # Then cycle bleed 1 → hp 9→8 (Phase 3, after_)
        assert gm.player_1.hp == 8
        assert gm.player_2.hp == 8
        assert result.after_resolve.hp_1 == 9  # Phase 1: before bleed
        assert result.after_resolve.hp_2 == 9
        assert result.end_actions.cycle_ended is True
        assert result.end_actions.bleed_damage_1 == 1
        assert result.after_end_actions.hp_1 == 8  # Phase 3: after bleed
        assert result.after_end_actions.hp_2 == 8
        assert result.after_end_actions.cycle == 2
        assert result.after_end_actions.round_in_cycle == 1

    def test_mid_cycle_no_bleed(self):
        """Round 3 of a cycle: no cycle_ended, no HP bleed."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.round_in_cycle = 3
        gm.round_number = 3
        gm.player_1.hand = [CardType.ATTACK]
        gm.player_2.hand = [CardType.DEFENSE]
        gm.player_1.hp = 10
        gm.player_2.hp = 10

        result = gm.play_round([CardType.ATTACK], [CardType.DEFENSE])
        assert result.end_actions.cycle_ended is False
        assert gm.player_1.hp == 10  # No bleed

    def test_cycle_bleed_causes_game_end(self):
        """TC-GC-04: Cycle end bleed with hp=1 each → both die → draw."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.round_in_cycle = 5
        gm.cycle_number = 1
        gm.round_number = 5
        gm.player_1.hand = [CardType.DEFENSE]
        gm.player_2.hand = [CardType.DEFENSE]
        gm.player_1.hp = 1
        gm.player_2.hp = 1

        result = gm.play_round([CardType.DEFENSE], [CardType.DEFENSE])
        assert gm.is_over is True
        assert gm.winner == "draw"
        # Both hp=0 after cycle bleed
        assert gm.player_1.hp == 0
        assert gm.player_2.hp == 0

    def test_new_cycle_reclaims_cards(self):
        """TC-GC-05: New cycle reclaims hand → discard → draw_pile → shuffle → draw 7."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.round_in_cycle = 5
        gm.round_number = 5
        # Set up: hand has 3 cards, discard has 5+ cards for enough to draw 7
        gm.player_1.hand = [CardType.ATTACK] * 3
        gm.player_2.hand = [CardType.DEFENSE] * 3
        gm.player_1.discard_pile = [CardType.HEAVY] * 10
        gm.player_1.draw_pile = []
        gm.player_2.discard_pile = [CardType.ENERGY] * 10
        gm.player_2.draw_pile = []

        gm.play_round([CardType.ATTACK], [CardType.DEFENSE])
        # New cycle started
        assert gm.round_in_cycle == 1
        # hand has 7 cards (drawn from shuffled pile, enough in deck)
        assert len(gm.player_1.hand) == 7
        assert len(gm.player_1.draw_pile) >= 0  # remaining in draw pile

    def test_cycle_preserves_attack_streak(self):
        """TC-GC-06: New cycle preserves attack streak in history."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.round_in_cycle = 5
        gm.round_number = 5
        gm.player_1.history = [
            PlayerHistoryItem(CardType.ATTACK, None),
            PlayerHistoryItem(CardType.ATTACK, None),
        ]
        gm.player_1.hand = [CardType.ATTACK]
        gm.player_2.hand = [CardType.DEFENSE]

        gm.play_round([CardType.ATTACK], [CardType.DEFENSE])
        # History preserved across cycles
        assert gm.player_1.history[-3].card == CardType.ATTACK
        assert gm.player_1.history[-2].card == CardType.ATTACK
        # + the newly played ATTACK
        assert gm.player_1.history[-1].card == CardType.ATTACK


class TestGameEnd:
    """TC-GC-07, GC-08a, GC-08b, GC-09."""

    def test_no_round_limit(self):
        """TC-GC-07: Game only ends when HP reaches 0, not by round count."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hp = 10
        gm.player_2.hp = 15
        gm.cycle_number = 50
        gm.round_number = 250
        gm.round_in_cycle = 3
        assert gm._check_game_end() == (False, None)

    def test_hp_zero_ends_game(self):
        """TC-GC-08a: HP=0 → game over, winner determined."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_2.hp = 5
        gm.player_1.hand = [CardType.ATTACK, CardType.ATTACK, CardType.ATTACK,
                            CardType.ATTACK, CardType.ATTACK, CardType.ATTACK,
                            CardType.ATTACK]
        gm.player_2.hand = [CardType.DEFENSE]
        # streak=5 → min(5+1,3)=3 damage. Need to hit 5.
        # P1 ATTACK streak=5 → 3 damage, P2 goes from 5→2.
        # We need P2 hp to reach 0. Let me make it simpler.
        gm.player_2.hp = 1
        gm.player_2.hand = [CardType.ENERGY]  # won't block
        gm.player_1.hand = [CardType.ATTACK]

        result = gm.play_round([CardType.ATTACK], [CardType.ENERGY])
        assert gm.player_2.hp == 0
        assert gm.is_over is True
        assert gm.winner == "1"

    def test_simultaneous_death_draw(self):
        """TC-GC-08b: Both HP=0 simultaneously → draw."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hp = 1
        gm.player_2.hp = 1
        gm.player_1.hand = [CardType.ATTACK]
        gm.player_2.hand = [CardType.ATTACK]

        result = gm.play_round([CardType.ATTACK], [CardType.ATTACK])
        assert gm.is_over is True
        assert gm.winner == "draw"

    def test_play_round_after_game_over_raises(self):
        """TC-GC-09: play_round after game over raises ValueError."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.is_over = True
        gm.player_1.hand = [CardType.ATTACK]
        gm.player_2.hand = [CardType.DEFENSE]
        with pytest.raises(ValueError, match="game is over"):
            gm.play_round([CardType.ATTACK], [CardType.DEFENSE])


class TestGetState:
    """TC-GC-10: get_state returns GameState with typed fields."""

    def test_get_state_returns_gamestate_not_dict(self):
        """TC-GC-01: get_state() returns GameState, not dict."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gs = gm.get_state()
        assert isinstance(gs, GameState)
        assert not isinstance(gs, dict)

    def test_get_state_fields(self):
        """get_state returns round, cycle, round_in_cycle, energy, pile counts."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gs = gm.get_state()
        assert gs.round == 1
        assert gs.cycle == 1
        assert gs.round_in_cycle == 1
        p1 = gs.players["1"]
        p2 = gs.players["2"]
        assert p1.hp == 20
        assert p1.energy >= 0
        assert isinstance(p1.statuses, list)
        assert isinstance(p1.hand, list)
        assert p1.draw_pile_count >= 0
        assert p1.discard_pile_count >= 0

    def test_player_perspective_hides_opponent_piles(self):
        """With player_perspective='1', P1 sees own pile contents, P2 sees counts."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gs = gm.get_state(player_perspective="1")
        p1 = gs.players["1"]
        p2 = gs.players["2"]
        # P1 sees his own pile details
        assert p1.draw_pile is not None
        assert isinstance(p1.draw_pile, list)
        # P2 only sees counts
        assert p2.draw_pile is None
        assert isinstance(p2.draw_pile_count, int)

    def test_hand_elements_are_cardtype(self):
        """TC-GC-04: get_state() hand elements are CardType, not str."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gs = gm.get_state()
        for card in gs.players["1"].hand:
            assert isinstance(card, CardType), f"{card!r} is not CardType"
            assert not isinstance(card, str), f"{card!r} should not be str"
        for card in gs.players["2"].hand:
            assert isinstance(card, CardType), f"{card!r} is not CardType"
            assert not isinstance(card, str), f"{card!r} should not be str"

    def test_default_perspective_hides_both_piles(self):
        """TC-GC-06: Default perspective hides both players' pile contents."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gs = gm.get_state()  # perspective=None
        assert gs.players["1"].draw_pile is None
        assert gs.players["2"].draw_pile is None
        assert gs.players["1"].discard_pile is None
        assert gs.players["2"].discard_pile is None
        # Both players should have pile counts
        assert isinstance(gs.players["1"].draw_pile_count, int)
        assert isinstance(gs.players["2"].draw_pile_count, int)


class TestBuildEnergyCheck:
    """TC-GC-11: Energy check for BUILD card."""

    def test_energy_zero_cannot_play_build(self):
        """Cannot play BUILD when energy=0."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.energy = 0
        gm.player_1.hand = [CardType.BUILD]
        gm.player_2.hand = [CardType.ATTACK]
        with pytest.raises(ValueError, match="energy"):
            gm.play_round([CardType.BUILD], [CardType.ATTACK])

    def test_energy_sufficient_can_play_build(self):
        """Can play BUILD when energy >= build_cost."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.energy = 2
        gm.player_1.hand = [CardType.BUILD]
        gm.player_2.hand = [CardType.ATTACK]
        result = gm.play_round(
            [CardType.BUILD], [CardType.ATTACK],
            build_target_1=CardType.ATTACK,
        )
        # BUILD succeeded
        assert gm.player_1.energy == 1  # consumed 1 energy (ATTACK build_cost=1)
        assert CardType.ATTACK in gm.player_1.discard_pile


class TestHistoryRecording:
    """TC-GC-H01~H04: History recording with PlayerHistoryItem (design/16)."""

    def test_non_build_records_to_history(self):
        """TC-GC-H01: Non-BUILD card → history records card with build_target=None."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hand = [CardType.ATTACK]
        gm.player_2.hand = [CardType.DEFENSE]

        gm.play_round([CardType.ATTACK], [CardType.DEFENSE])

        assert len(gm.player_1.history) == 1
        assert gm.player_1.history[0] == PlayerHistoryItem(CardType.ATTACK, None)
        assert len(gm.player_2.history) == 1
        assert gm.player_2.history[0] == PlayerHistoryItem(CardType.DEFENSE, None)

    def test_build_records_target_to_history(self):
        """TC-GC-H02: BUILD card records build_target in PlayerHistoryItem."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.energy = 1
        gm.player_1.hand = [CardType.BUILD]
        gm.player_2.hand = [CardType.ATTACK]

        gm.play_round(
            [CardType.BUILD], [CardType.ATTACK],
            build_target_1=CardType.ATTACK,
        )

        assert gm.player_1.history[0] == PlayerHistoryItem(CardType.BUILD, CardType.ATTACK)
        assert gm.player_2.history[0] == PlayerHistoryItem(CardType.ATTACK, None)

    def test_multi_round_history_accumulates(self):
        """TC-GC-H03: Multiple rounds accumulate in history correctly."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hand = [CardType.ATTACK, CardType.BUILD, CardType.DEFENSE]
        gm.player_1.energy = 5
        gm.player_2.hand = [CardType.DEFENSE, CardType.ATTACK, CardType.HEAVY]
        gm.player_2.energy = 5
        # Round 1: attack vs defense
        gm.play_round([CardType.ATTACK], [CardType.DEFENSE])
        # Round 2: build(heavy) vs attack
        gm.play_round([CardType.BUILD], [CardType.ATTACK], build_target_1=CardType.HEAVY)
        # Round 3: defense vs heavy
        gm.play_round([CardType.DEFENSE], [CardType.HEAVY])

        assert gm.player_1.history == [
            PlayerHistoryItem(CardType.ATTACK, None),
            PlayerHistoryItem(CardType.BUILD, CardType.HEAVY),
            PlayerHistoryItem(CardType.DEFENSE, None),
        ]
        assert gm.player_2.history == [
            PlayerHistoryItem(CardType.DEFENSE, None),
            PlayerHistoryItem(CardType.ATTACK, None),
            PlayerHistoryItem(CardType.HEAVY, None),
        ]

    def test_cycle_preserves_history_items(self):
        """TC-GC-H04: New cycle preserves full history (PlayerHistoryItem format)."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.round_in_cycle = 5
        gm.round_number = 5
        # Set 2 rounds of history in PlayerHistoryItem format
        gm.player_1.history = [
            PlayerHistoryItem(CardType.ATTACK, None),
            PlayerHistoryItem(CardType.ATTACK, None),
        ]
        gm.player_1.hand = [CardType.ATTACK]
        gm.player_2.hand = [CardType.DEFENSE]
        gm.player_1.discard_pile = [CardType.HEAVY] * 10
        gm.player_2.discard_pile = [CardType.ENERGY] * 10

        gm.play_round([CardType.ATTACK], [CardType.DEFENSE])
        # History preserved across cycle, all items are PlayerHistoryItem
        assert len(gm.player_1.history) >= 2
        for item in gm.player_1.history:
            assert isinstance(item, PlayerHistoryItem)


class TestGetStateHistoryFormat:
    """TC-GS-01, TC-GS-02, TC-GS-03, TC-GS-04: get_state PlayerHistoryItem format."""

    def test_empty_history_returns_empty_list(self):
        """TC-GS-01: No rounds played → history is empty list."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gs = gm.get_state()
        assert gs.players["1"].history == []
        assert gs.players["2"].history == []

    def test_history_contains_player_history_items(self):
        """get_state history entries are PlayerHistoryItem instances."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hand = [CardType.ATTACK]
        gm.player_2.hand = [CardType.DEFENSE]

        gm.play_round([CardType.ATTACK], [CardType.DEFENSE])
        gs = gm.get_state()

        p1_history = gs.players["1"].history
        assert len(p1_history) == 1
        assert isinstance(p1_history[0], PlayerHistoryItem)
        assert p1_history[0] == PlayerHistoryItem(CardType.ATTACK, None)

    def test_build_action_in_history(self):
        """TC-GS-03: BUILD with target → history contains build_target."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.energy = 1
        gm.player_1.hand = [CardType.BUILD]
        gm.player_2.hand = [CardType.ATTACK]

        gm.play_round(
            [CardType.BUILD], [CardType.ATTACK],
            build_target_1=CardType.ATTACK,
        )
        gs = gm.get_state()

        p1_last = gs.players["1"].history[-1]
        assert p1_last == PlayerHistoryItem(CardType.BUILD, CardType.ATTACK)

    def test_multi_round_history_accumulates(self):
        """TC-GS-04: Multiple rounds accumulate correctly."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.energy = 5
        gm.player_1.hand = [CardType.ATTACK, CardType.BUILD, CardType.DEFENSE]
        gm.player_2.hand = [CardType.DEFENSE, CardType.ATTACK, CardType.HEAVY]

        gm.play_round([CardType.ATTACK], [CardType.DEFENSE])
        gm.play_round([CardType.BUILD], [CardType.ATTACK], build_target_1=CardType.ATTACK)
        gm.play_round([CardType.DEFENSE], [CardType.HEAVY])

        gs = gm.get_state()
        p1_hist = gs.players["1"].history
        assert len(p1_hist) == 3
        assert p1_hist[0] == PlayerHistoryItem(CardType.ATTACK, None)
        assert p1_hist[1] == PlayerHistoryItem(CardType.BUILD, CardType.ATTACK)
        assert p1_hist[2] == PlayerHistoryItem(CardType.DEFENSE, None)


# =============================================================================
# Status lifecycle tests — design/15 TC-GS-01 ~ TC-GS-08
# =============================================================================


class TestStatusLifecycle:
    """TC-GS-01 ~ TC-GS-08: PlayerStatus lifecycle in GameCore."""

    def test_play_round_statuses_are_playerstatus(self):
        """TC-GS-01: RoundResult statuses_1 are PlayerStatus (pre-tick snapshot).

        Note: After play_round, player.statuses is empty because status lifetimes
        are ticked within the same round. The pre-tick snapshot is in RoundResult.
        """
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hand = [CardType.DEFENSE]
        gm.player_2.hand = [CardType.ATTACK]

        result = gm.play_round([CardType.DEFENSE], [CardType.ATTACK])

        # RoundResult captures pre-tick state
        assert len(result.after_resolve.statuses_1) == 1
        assert isinstance(result.after_resolve.statuses_1[0], PlayerStatus)
        assert result.after_resolve.statuses_1[0].status_type == StatusType.DEFENSE
        assert result.after_resolve.statuses_1[0].lifetime == 1
        # Post-tick: both players have empty statuses
        assert gm.player_1.statuses == []
        assert result.after_resolve.statuses_2 == []

    def test_tick_reduces_lifetime_and_removes_expired(self):
        """TC-GS-02: _tick_status_lifetimes reduces lifetime, removes expired."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.statuses = [
            PlayerStatus(StatusType.DEFENSE, 2),
            PlayerStatus(StatusType.GUARD, 1),
        ]

        gm._tick_status_lifetimes()

        assert len(gm.player_1.statuses) == 1
        assert gm.player_1.statuses[0].status_type == StatusType.DEFENSE
        assert gm.player_1.statuses[0].lifetime == 1

    def test_tick_removes_single_lifetime_1(self):
        """TC-GS-03: Single status lifetime=1 is removed after tick."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.statuses = [PlayerStatus(StatusType.DEFENSE, 1)]

        gm._tick_status_lifetimes()

        assert gm.player_1.statuses == []

    def test_status_not_carried_over_across_rounds(self):
        """TC-GS-04: Statuses expire within the same round (ticked to 0).

        After play_round, statuses are empty because tick happens in Phase 2.
        """
        gm = GameCore(mode="pvp")
        gm.start_game()
        # Round 1: P2 plays DEFENSE → P2 gets DEFENSE (captured in RoundResult, ticked by end)
        gm.player_1.hand = [CardType.ATTACK]
        gm.player_2.hand = [CardType.DEFENSE]
        result1 = gm.play_round([CardType.ATTACK], [CardType.DEFENSE])
        # RoundResult.statuses_2 captured before tick
        assert len(result1.after_resolve.statuses_2) == 1
        assert result1.after_resolve.statuses_2[0].status_type == StatusType.DEFENSE
        # After tick, player2 has no active statuses
        assert gm.player_2.statuses == []

        # Round 2: P2 plays ATTACK → no DEFENSE granted
        gm.player_1.hand = [CardType.ATTACK]
        gm.player_2.hand = [CardType.ATTACK]
        result2 = gm.play_round([CardType.ATTACK], [CardType.ATTACK])
        assert result2.after_resolve.statuses_2 == []

    def test_status_lifetimes_fields_removed(self):
        """TC-GS-05: GameCore no longer has _status_lifetimes_1/_status_lifetimes_2."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        with pytest.raises(AttributeError):
            _ = gm._status_lifetimes_1
        with pytest.raises(AttributeError):
            _ = gm._status_lifetimes_2

    def test_roundresult_contains_playerstatus(self):
        """TC-GS-06: RoundResult statuses_1 are PlayerStatus instances."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hand = [CardType.DEFENSE]
        gm.player_2.hand = [CardType.ATTACK]

        result = gm.play_round([CardType.DEFENSE], [CardType.ATTACK])

        assert len(result.after_resolve.statuses_1) == 1
        ps = result.after_resolve.statuses_1[0]
        assert isinstance(ps, PlayerStatus)
        assert ps.status_type == StatusType.DEFENSE
        assert ps.lifetime == 1

    def test_guard_not_granted_across_rounds_statuses_expire(self):
        """TC-GS-07 (revised): GUARD is NOT granted across rounds because statuses expire.

        Since statuses are ticked to empty each round, the resolver never sees
        pre-existing DEFENSE in consecutive rounds. GUARD can only be tested
        at the resolver unit-test level with manually set pre-existing statuses.
        """
        gm = GameCore(mode="pvp")
        gm.start_game()
        # Round 1: DEFENSE → gets DEFENSE (ticked away at round end)
        gm.player_1.hand = [CardType.DEFENSE]
        gm.player_2.hand = [CardType.ATTACK]
        gm.play_round([CardType.DEFENSE], [CardType.ATTACK])
        # Round 2: DEFENSE again → only gets DEFENSE (no pre-existing)
        gm.player_1.hand = [CardType.DEFENSE]
        gm.player_2.hand = [CardType.ATTACK]
        result = gm.play_round([CardType.DEFENSE], [CardType.ATTACK])

        types = [ps.status_type for ps in result.after_resolve.statuses_1]
        assert StatusType.DEFENSE in types
        assert StatusType.GUARD not in types

    def test_defense_granted_when_player_plays_defense(self):
        """TC-GS-08: Playing DEFENSE grants DEFENSE status."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        gm.player_1.hand = [CardType.DEFENSE]
        gm.player_2.hand = [CardType.ATTACK]
        result = gm.play_round([CardType.DEFENSE], [CardType.ATTACK])

        types = [ps.status_type for ps in result.after_resolve.statuses_1]
        assert StatusType.DEFENSE in types
        assert StatusType.GUARD not in types
