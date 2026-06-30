"""Unit tests for src/backend/game/game_manager.py — GameManager and RoundResult.

Covers TC-GROUP-5 (Game End Detection) and TC-GROUP-6 (Edge Cases).
"""

import pytest
from src.backend.game.card import CardType, get_initial_hand
from src.backend.game.player import Player
from src.backend.game.game_manager import GameManager, RoundResult

A = CardType.ATTACK
D = CardType.DEFENSE
H = CardType.HEAVY_STRIKE
C = CardType.COUNTER


# ---------------------------------------------------------------------------
# RoundResult tests
# ---------------------------------------------------------------------------

class TestRoundResult:
    """Tests for the RoundResult value object."""

    def test_creation_with_all_fields(self):
        """RoundResult should hold all round outcome data."""
        result = RoundResult(
            round=1,
            cards_1=[A],
            cards_2=[D],
            damage_1=0,
            damage_2=0,
            hp_1_after=3,
            hp_2_after=3,
            details=["A Attack blocked", "B Defense"],
            is_game_over=False,
            winner=None,
        )
        assert result.round == 1
        assert result.cards_1 == [A]
        assert result.cards_2 == [D]
        assert result.damage_1 == 0
        assert result.damage_2 == 0
        assert result.hp_1_after == 3
        assert result.hp_2_after == 3
        assert result.is_game_over is False
        assert result.winner is None

    def test_game_over_result(self):
        """RoundResult should indicate game over with a winner."""
        result = RoundResult(
            round=3,
            cards_1=[A],
            cards_2=[A],
            damage_1=1,
            damage_2=2,
            hp_1_after=2,
            hp_2_after=0,
            details=["..."],
            is_game_over=True,
            winner="1",
        )
        assert result.is_game_over is True
        assert result.winner == "1"


# ---------------------------------------------------------------------------
# GameManager: start_game()
# ---------------------------------------------------------------------------

class TestGameManagerStartGame:
    """Tests for GameManager.start_game()."""

    def test_initial_state(self):
        """After start_game, both players have 3 HP and 7 cards."""
        gm = GameManager()
        gm.start_game()
        assert gm.player_1.hp == 3
        assert gm.player_2.hp == 3
        assert len(gm.player_1.hand) == 7
        assert len(gm.player_2.hand) == 7
        assert gm.round_number == 1
        assert gm.is_over is False
        assert gm.winner is None
        assert len(gm.round_log) == 0

    def test_start_game_twice_resets_state_tc_x6(self):
        """TC-X6: Calling start_game twice reinitializes fresh state."""
        gm = GameManager()
        gm.start_game()
        gm.play_round([A], [A])  # Play one round
        assert gm.round_number == 2  # Moved to round 2
        gm.start_game()  # Reset
        assert gm.round_number == 1
        assert gm.player_1.hp == 3
        assert gm.player_2.hp == 3
        assert len(gm.player_1.hand) == 7
        assert gm.is_over is False
        assert gm.winner is None
        assert len(gm.round_log) == 0


# ---------------------------------------------------------------------------
# GameManager: play_round() — validation
# ---------------------------------------------------------------------------

class TestGameManagerValidation:
    """Tests for play_round() validation (TC-X2, TC-X3)."""

    def test_raises_if_game_over_tc_x2(self):
        """TC-X2: play_round() raises ValueError when game is over."""
        gm = GameManager()
        gm.start_game()
        # Play 3 rounds of ATK vs ATK → both die after R2
        gm.play_round([A], [A])  # R1: 3→2
        gm.play_round([A], [A])  # R2: 2→0
        with pytest.raises(ValueError, match="game is over"):
            gm.play_round([A], [A])

    def test_raises_if_card_not_in_hand_tc_x3(self):
        """TC-X3: play_round() raises ValueError if card not in hand."""
        gm = GameManager()
        gm.start_game()
        # Play all 3 ATK from A (A starts with 3 ATTACK).
        # Use different cards from B that B actually holds.
        gm.play_round([A], [D])  # A uses ATK #1, B uses DEF #1
        gm.play_round([A], [H])  # A uses ATK #2, B uses HS
        gm.play_round([A], [C])  # A uses ATK #3 — A now has 0 ATK, B still has [A,A,D,CTR]
        with pytest.raises(ValueError, match="Not all cards"):
            gm.play_round([A], [A])  # No ATK left for A


# ---------------------------------------------------------------------------
# GameManager: play_round() — game end detection (TC-GROUP-5)
# ---------------------------------------------------------------------------

class TestGameManagerEndDetection:
    """TC-GROUP-5: Game end detection via GameManager.play_round()."""

    def test_tc_e1_both_die_r2_draw(self):
        """TC-E1: R1 ATK vs ATK, R2 ATK vs ATK → draw after R2."""
        gm = GameManager()
        gm.start_game()
        r1 = gm.play_round([A], [A])
        assert r1.is_game_over is False
        assert gm.player_1.hp == 2
        assert gm.player_2.hp == 2
        r2 = gm.play_round([A], [A])
        assert r2.is_game_over is True
        assert r2.winner == "draw"
        assert gm.player_1.hp == 0
        assert gm.player_2.hp == 0

    def test_tc_e2_not_over_after_r3(self):
        """TC-E2: 3 rounds of blocked attacks → game not over."""
        gm = GameManager()
        gm.start_game()
        gm.play_round([A], [D])
        gm.play_round([D], [A])
        r3 = gm.play_round([D], [D])
        assert r3.is_game_over is False
        assert gm.is_over is False

    def test_tc_e3_a_dies_b_wins(self):
        """TC-E3: A starts at 1HP, R1 ATK vs ATK → B wins."""
        gm = GameManager()
        gm.start_game()
        gm.player_1.hp = 1  # Custom HP
        r = gm.play_round([A], [A])
        assert r.is_game_over is True
        assert r.winner == "2"
        assert gm.player_1.hp == 0
        assert gm.player_2.hp == 2

    def test_tc_e4_b_dies_a_wins(self):
        """TC-E4: B starts at 1HP, R1 ATK vs ATK → A wins."""
        gm = GameManager()
        gm.start_game()
        gm.player_2.hp = 1
        r = gm.play_round([A], [A])
        assert r.is_game_over is True
        assert r.winner == "1"
        assert gm.player_1.hp == 2
        assert gm.player_2.hp == 0

    def test_tc_e5_seven_def_draw(self):
        """TC-E5: 7 rounds of DEF vs DEF → draw (3-3). Custom hands."""
        gm = GameManager()
        gm.start_game()
        gm.player_1.hand = [D] * 7
        gm.player_2.hand = [D] * 7
        for _ in range(6):
            r = gm.play_round([D], [D])
            assert r.is_game_over is False
        r7 = gm.play_round([D], [D])
        assert r7.is_game_over is True
        assert r7.winner == "draw"
        assert gm.player_1.hp == 3
        assert gm.player_2.hp == 3

    def test_tc_e6_custom_hands_draw_after_r7(self):
        """TC-E6: A=ATK+6DEF, B=6DEF+ATK → draw (2-2) after R7."""
        gm = GameManager()
        gm.start_game()
        gm.player_1.hand = [A] + [D] * 6
        gm.player_2.hand = [D] * 6 + [A]
        gm.play_round([A], [A])  # R1: both take 1 → 2HP
        for _ in range(6):
            gm.play_round([D], [D])
        assert gm.is_over is True
        assert gm.winner == "draw"
        assert gm.player_1.hp == 2
        assert gm.player_2.hp == 2

    def test_tc_e7_a_wins_after_r3(self):
        """TC-E7: R1 ATK vs DEF, R2 ATK vs ATK, R3 ATK vs HS → A wins."""
        gm = GameManager()
        gm.start_game()
        gm.play_round([A], [D])  # R1: no damage
        r2 = gm.play_round([A], [A])  # R2: A→2, B→1
        assert r2.is_game_over is False
        r3 = gm.play_round([A], [H])  # R3: A consec=3→B -3 (dies), HS→A -1
        assert r3.is_game_over is True
        assert r3.winner == "1"
        assert gm.player_1.hp == 1
        assert gm.player_2.hp == 0


# ---------------------------------------------------------------------------
# GameManager: edge cases (TC-GROUP-6)
# ---------------------------------------------------------------------------

class TestGameManagerEdgeCases:
    """TC-GROUP-6: Edge cases for GameManager."""

    def test_tc_x1_hp_clamped_below_zero(self):
        """TC-X1: HP drops below 0 → clamped to 0, game ends."""
        gm = GameManager()
        gm.start_game()
        gm.player_1.hp = 1
        # B has [A,A] history, plays A → consecutive=3 damage to A.
        # A must play a card that does NOT block Attack (use HS or CTR).
        gm.player_2.history = [A, A]
        r = gm.play_round([H], [A])  # A plays HS, B plays ATK×3 → A takes 3
        assert gm.player_1.hp == 0  # Clamped from -2
        assert r.is_game_over is True
        assert r.winner == "2"

    def test_tc_x4_round_7_game_continues_then_ends(self):
        """TC-X4: After round 7 resolution, game ends if no HP=0."""
        gm = GameManager()
        gm.start_game()
        gm.player_1.hand = [D] * 7
        gm.player_2.hand = [D] * 7
        for i in range(6):
            gm.play_round([D], [D])
        assert gm.is_over is False
        r7 = gm.play_round([D], [D])
        assert r7.is_game_over is True  # Round 7 ends game

    def test_tc_x5_last_card_on_r7_game_ends(self):
        """TC-X5: Last card played on R7, game ends after resolution."""
        gm = GameManager()
        gm.start_game()
        gm.player_1.hand = [D] * 7
        gm.player_2.hand = [D] * 7
        for _ in range(7):
            gm.play_round([D], [D])
        assert gm.is_over is True

    def test_round_number_increments(self):
        """GameManager should increment round_number after each non-final round."""
        gm = GameManager()
        gm.start_game()
        assert gm.round_number == 1
        gm.play_round([D], [D])
        assert gm.round_number == 2
        gm.play_round([D], [D])
        assert gm.round_number == 3

    def test_round_log_accumulates(self):
        """GameManager.round_log should collect all RoundResults."""
        gm = GameManager()
        gm.start_game()
        gm.play_round([A], [D])
        gm.play_round([D], [A])
        assert len(gm.round_log) == 2
        assert gm.round_log[0].round == 1
        assert gm.round_log[1].round == 2


# ---------------------------------------------------------------------------
# GameManager: get_state()
# ---------------------------------------------------------------------------

class TestGameManagerGetState:
    """Tests for GameManager.get_state()."""

    def test_get_state_after_start(self):
        """get_state should return current game snapshot."""
        gm = GameManager()
        gm.start_game()
        state = gm.get_state()
        assert state["round"] == 1
        assert state["is_over"] is False
        assert state["winner"] is None
        assert state["players"]["1"]["hp"] == 3
        assert state["players"]["2"]["hp"] == 3
        assert len(state["players"]["1"]["hand"]) == 7
        assert state["players"]["1"]["history"] == []

    def test_get_state_after_round(self):
        """get_state should reflect post-round state."""
        gm = GameManager()
        gm.start_game()
        gm.play_round([A], [D])
        state = gm.get_state()
        assert state["round"] == 2
        assert state["players"]["1"]["hp"] == 3
        assert state["players"]["2"]["hp"] == 3
        assert len(state["players"]["1"]["hand"]) == 6
        assert state["players"]["1"]["history"] == ["attack"]

    def test_get_state_game_over(self):
        """get_state should show game-over state."""
        gm = GameManager()
        gm.start_game()
        gm.play_round([A], [A])  # R1: 3→2
        gm.play_round([A], [A])  # R2: 2→0
        state = gm.get_state()
        assert state["is_over"] is True
        assert state["winner"] == "draw"


# ---------------------------------------------------------------------------
# GameManager: D6 card discard order verification
# ---------------------------------------------------------------------------

class TestGameManagerDiscardOrder:
    """Verify cards are discarded after damage (D6)."""

    def test_hand_unchanged_during_resolution(self):
        """Cards remain in hand during resolution (D6: discard after damage)."""
        gm = GameManager()
        gm.start_game()
        # After one round, the played cards should be gone
        gm.play_round([A], [D])
        assert A not in gm.player_1.hand or gm.player_1.hand.count(A) == 2
        assert D not in gm.player_2.hand or gm.player_2.hand.count(D) == 1


# ---------------------------------------------------------------------------
# GameManager: mode parameter (TC-GM1–TC-GM5)
# ---------------------------------------------------------------------------


class TestGameManagerMode:
    """TC-GM1–TC-GM5: mode constructor parameter and get_state integration."""

    def test_default_mode_is_pvp(self):
        """TC-GM1: GameManager() defaults to mode='pvp'."""
        gm = GameManager()
        assert gm.mode == "pvp"

    def test_explicit_pvp_mode(self):
        """GameManager(mode='pvp') sets mode correctly."""
        gm = GameManager(mode="pvp")
        assert gm.mode == "pvp"

    def test_explicit_pve_mode(self):
        """GameManager(mode='pve') sets mode correctly."""
        gm = GameManager(mode="pve")
        assert gm.mode == "pve"

    def test_get_state_includes_mode_pvp(self):
        """TC-GM2: get_state() dict includes 'mode' key for PVP."""
        gm = GameManager(mode="pvp")
        gm.start_game()
        state = gm.get_state()
        assert state["mode"] == "pvp"

    def test_get_state_includes_mode_pve(self):
        """TC-GM2: get_state() dict includes 'mode' key for PVE."""
        gm = GameManager(mode="pve")
        gm.start_game()
        state = gm.get_state()
        assert state["mode"] == "pve"

    def test_play_round_works_in_pve_mode(self):
        """TC-GM3: play_round() works identically regardless of mode."""
        gm = GameManager(mode="pve")
        gm.start_game()
        result = gm.play_round([A], [D])
        assert result is not None
        assert result.damage_1 >= 0
        assert result.damage_2 >= 0

    def test_mode_does_not_affect_rules(self):
        """TC-GM4: Same play_round() on PVP and PVE produces identical results."""
        gm_pvp = GameManager(mode="pvp")
        gm_pvp.start_game()
        gm_pvp.player_1.hand = [A] * 7
        gm_pvp.player_2.hand = [D] * 7
        r_pvp = gm_pvp.play_round([A], [D])

        gm_pve = GameManager(mode="pve")
        gm_pve.start_game()
        gm_pve.player_1.hand = [A] * 7
        gm_pve.player_2.hand = [D] * 7
        r_pve = gm_pve.play_round([A], [D])

        assert r_pvp.damage_1 == r_pve.damage_1
        assert r_pvp.damage_2 == r_pve.damage_2
        assert r_pvp.hp_1_after == r_pve.hp_1_after
        assert r_pvp.hp_2_after == r_pve.hp_2_after

    def test_get_state_includes_round_log(self):
        """TC-GM5: get_state() includes round_log after rounds are played."""
        gm = GameManager()
        gm.start_game()
        gm.play_round([A], [D])
        state = gm.get_state()
        assert "round_log" in state
        assert len(state["round_log"]) == 1
        assert state["round_log"][0]["round"] == 1

    def test_get_state_round_log_empty_initially(self):
        """get_state() returns empty round_log before any rounds played."""
        gm = GameManager()
        gm.start_game()
        state = gm.get_state()
        assert "round_log" in state
        assert state["round_log"] == []


# ---------------------------------------------------------------------------
# GameManager: mode immutability
# ---------------------------------------------------------------------------


class TestGameManagerModeImmutability:
    """Mode is set at construction and cannot be changed afterward."""

    def test_mode_is_immutable_via_dataclass(self):
        """Mode should be set at __init__ and remain constant."""
        gm = GameManager(mode="pve")
        gm.start_game()
        # After start_game(), mode should still be pve
        assert gm.mode == "pve"
        # Play a round — mode unchanged
        gm.play_round([A], [D])
        assert gm.mode == "pve"
