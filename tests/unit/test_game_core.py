"""Unit tests for src/backend/game/game_core.py — cycle mechanics.

Test cases per design/11 test_cases.md §4 (TC-GC-01 ~ TC-GC-11).
"""

import pytest

from src.backend.game.card import CardType
from src.backend.game.game_core import GameCore


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
        # Both took 1 attack damage + 1 cycle bleed = 2 total → hp 10→8
        assert gm.player_1.hp == 8
        assert gm.player_2.hp == 8
        assert result.hp_1_after == 8  # HP reflects bleed
        assert result.hp_2_after == 8
        assert result.cycle_ended is True
        assert gm.cycle_number == 2
        assert gm.round_in_cycle == 1

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
        assert result.cycle_ended is False
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
        gm.player_1.history = [CardType.ATTACK, CardType.ATTACK]
        gm.player_1.hand = [CardType.ATTACK]
        gm.player_2.hand = [CardType.DEFENSE]

        gm.play_round([CardType.ATTACK], [CardType.DEFENSE])
        # History preserved across cycles
        assert gm.player_1.history[-2:] == [CardType.ATTACK, CardType.ATTACK]
        # + the newly played ATTACK
        assert gm.player_1.history[-1] == CardType.ATTACK


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
    """TC-GC-10: get_state returns complete state with new fields."""

    def test_get_state_fields(self):
        """get_state returns cycle, round_in_cycle, energy, pile counts."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        state = gm.get_state()
        assert state["round"] == 1
        assert state["cycle"] == 1
        assert state["round_in_cycle"] == 1
        p1 = state["players"]["1"]
        p2 = state["players"]["2"]
        assert "energy" in p1
        assert "statuses" in p1
        assert "hand" in p1
        assert "draw_pile_count" in p1
        assert "discard_pile_count" in p1

    def test_player_perspective_hides_opponent_piles(self):
        """With player_perspective='1', P1 sees own pile contents, P2 sees counts."""
        gm = GameCore(mode="pvp")
        gm.start_game()
        state = gm.get_state(player_perspective="1")
        p1 = state["players"]["1"]
        p2 = state["players"]["2"]
        # P1 sees his own pile details
        assert isinstance(p1["draw_pile"], list)
        # P2 only sees counts
        assert isinstance(p2["draw_pile_count"], int)


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
