"""Integration tests for the custom card game.

Tests full multi-round game flows that exercise the entire stack:
CardType → Player → CardResolver → GameManager.

These tests verify that all modules work together correctly across
complete game playthroughs, covering scenarios from test_cases.md
that cross module boundaries.
"""

import pytest
from src.backend.game.card import CardType, get_initial_hand
from src.backend.game.player import Player
from src.backend.game.resolver import CardResolver, ResolutionResult
from src.backend.game.game_manager import GameManager, RoundResult

# Short aliases
A = CardType.ATTACK
D = CardType.DEFENSE
H = CardType.HEAVY_STRIKE
C = CardType.COUNTER


# ---------------------------------------------------------------------------
# Full game playthroughs
# ---------------------------------------------------------------------------

class TestFullGamePlaythroughs:
    """Complete game scenarios from start to end."""

    def test_draw_after_seven_defensive_rounds_custom(self):
        """Full game: 7 rounds of DEF vs DEF → draw (3-3 HP)."""
        gm = GameManager()
        gm.start_game()
        gm.player_a.hand = [D] * 7
        gm.player_b.hand = [D] * 7
        for rnd in range(1, 8):
            result = gm.play_round(D, D)
            assert result.damage_a == 0
            assert result.damage_b == 0
            assert result.hp_a_after == 3
            assert result.hp_b_after == 3
            if rnd < 7:
                assert result.is_game_over is False
                assert gm.is_over is False
            else:
                assert result.is_game_over is True
                assert result.winner == "draw"
        assert gm.is_over is True
        assert gm.winner == "draw"
        assert len(gm.round_log) == 7

    def test_quick_draw_three_attack_rounds(self):
        """Full game: 3 rounds ATK vs ATK → draw after R2 (both HP→0)."""
        gm = GameManager()
        gm.start_game()
        # R1: both take 1 → HP 2
        r1 = gm.play_round(A, A)
        assert r1.damage_a == 1
        assert r1.damage_b == 1
        assert r1.hp_a_after == 2
        assert r1.hp_b_after == 2
        assert r1.is_game_over is False
        # R2: consecutive=2 → both take 2 → HP 0
        r2 = gm.play_round(A, A)
        assert r2.damage_a == 2
        assert r2.damage_b == 2
        assert r2.hp_a_after == 0
        assert r2.hp_b_after == 0
        assert r2.is_game_over is True
        assert r2.winner == "draw"
        assert gm.is_over is True

    def test_a_wins_by_hp_advantage_after_7_rounds(self):
        """Full game: A leads 3-2 after R1, 6 DEF rounds → A wins after R7."""
        gm = GameManager()
        gm.start_game()
        # R1: HS vs DEF (no prev) → B takes 1
        gm.play_round(H, D)
        assert gm.player_a.hp == 3
        assert gm.player_b.hp == 2
        # R2-R7: all DEF vs DEF with custom hands
        gm.player_a.hand = [D] * 6
        gm.player_b.hand = [D] * 6
        for _ in range(6):
            gm.play_round(D, D)
        assert gm.is_over is True
        assert gm.winner == "A"
        assert gm.player_a.hp == 3
        assert gm.player_b.hp == 2

    def test_game_with_all_card_types(self):
        """Full game using every card type at least once.
        
        R1: ATK vs DEF (0-0)
        R2: HS vs CTR (1-1) → HP 2-2
        R3: CTR vs ATK (0-1) → HP 2-1
        R4: ATK vs ATK (1-2) → HP 1-0 → A's ATK is consec=2, B's ATK is consec=1 → B dies, A wins
        """
        gm = GameManager()
        gm.start_game()
        
        r1 = gm.play_round(A, D)
        assert r1.damage_a == 0
        assert r1.damage_b == 0
        
        r2 = gm.play_round(H, C)
        assert r2.damage_a == 1
        assert r2.damage_b == 1
        assert gm.player_a.hp == 2
        assert gm.player_b.hp == 2
        
        r3 = gm.play_round(C, A)
        assert r3.damage_a == 0
        assert r3.damage_b == 1
        assert gm.player_a.hp == 2
        assert gm.player_b.hp == 1
        
        r4 = gm.play_round(A, A)
        # A history: [A, H, C, A] → trailing ATTACK streak = 1 (C breaks it) → dmg_b = 1
        # B history: [D, C, A, A] → trailing ATTACK streak = 2 → dmg_a = 2
        assert r4.damage_a == 2
        assert r4.damage_b == 1
        assert gm.player_a.hp == 0  # 2-2=0
        assert gm.player_b.hp == 0  # 1-1=0
        assert r4.is_game_over is True
        assert r4.winner == "draw"

    def test_counter_vs_counter_no_damage(self):
        """Multiple counter-vs-counter rounds produce no damage."""
        gm = GameManager()
        gm.start_game()
        gm.player_a.hand = [C] * 7
        gm.player_b.hand = [C] * 7
        for _ in range(7):
            r = gm.play_round(C, C)
            assert r.damage_a == 0
            assert r.damage_b == 0
        assert gm.is_over is True
        assert gm.winner == "draw"
        assert gm.player_a.hp == 3
        assert gm.player_b.hp == 3


# ---------------------------------------------------------------------------
# State consistency across rounds
# ---------------------------------------------------------------------------

class TestStateConsistency:
    """Verify game state remains consistent across rounds."""

    def test_hand_sizes_decrement_each_round(self):
        """Each round removes exactly one card from each player's hand."""
        gm = GameManager()
        gm.start_game()
        assert len(gm.player_a.hand) == 7
        assert len(gm.player_b.hand) == 7
        
        gm.play_round(A, D)
        assert len(gm.player_a.hand) == 6
        assert len(gm.player_b.hand) == 6
        
        gm.play_round(D, H)
        assert len(gm.player_a.hand) == 5
        assert len(gm.player_b.hand) == 5

    def test_history_grows_each_round(self):
        """Each round appends the played card to history."""
        gm = GameManager()
        gm.start_game()
        
        gm.play_round(A, D)
        assert gm.player_a.history == [A]
        assert gm.player_b.history == [D]
        
        gm.play_round(H, C)
        assert gm.player_a.history == [A, H]
        assert gm.player_b.history == [D, C]
        
        gm.play_round(C, A)
        assert gm.player_a.history == [A, H, C]
        assert gm.player_b.history == [D, C, A]

    def test_round_log_matches_round_number(self):
        """Round log entries match the round numbers they were played at."""
        gm = GameManager()
        gm.start_game()
        gm.player_a.hand = [D] * 7
        gm.player_b.hand = [D] * 7
        
        for rnd in range(1, 8):
            gm.play_round(D, D)
        assert len(gm.round_log) == 7
        for i, entry in enumerate(gm.round_log):
            assert entry.round == i + 1

    def test_hp_never_negative(self):
        """HP must never go below 0 regardless of damage (D5)."""
        gm = GameManager()
        gm.start_game()
        gm.player_a.hp = 1
        # B plays ATK with 3× consecutive → 3 damage to A (1 HP)
        gm.player_b.history = [A, A]
        r = gm.play_round(H, A)
        assert gm.player_a.hp == 0  # Clamped
        assert r.hp_a_after == 0
        assert gm.player_b.hp == 2  # A's HS deals 1

    def test_game_not_over_mid_sequence(self):
        """Game is not over after playing partial rounds."""
        gm = GameManager()
        gm.start_game()
        gm.player_a.hand = [D] * 7
        gm.player_b.hand = [D] * 7
        
        for rnd in range(1, 7):
            gm.play_round(D, D)
            assert gm.is_over is False
        assert gm.round_number == 7  # After 6 rounds, we're on round 7
        # Round 7 ends the game
        gm.play_round(D, D)
        assert gm.is_over is True

    def test_cannot_play_after_game_over(self):
        """play_round() raises ValueError after the game has ended."""
        gm = GameManager()
        gm.start_game()
        gm.player_a.hp = 1
        gm.player_b.hp = 1
        r = gm.play_round(A, A)
        assert r.is_game_over is True
        
        with pytest.raises(ValueError, match="game is over"):
            gm.play_round(D, D)

    def test_cannot_play_before_start(self):
        """play_round() raises ValueError if start_game() was not called."""
        gm = GameManager()
        with pytest.raises(ValueError, match="game not started"):
            gm.play_round(A, A)


# ---------------------------------------------------------------------------
# Resolution-GameManager bridge tests
# ---------------------------------------------------------------------------

class TestResolverGameManagerBridge:
    """Verify that CardResolver and GameManager interact correctly."""

    def test_resolver_damage_directly_reflected_in_hp(self):
        """Damage returned by CardResolver is exactly what's applied to HP."""
        gm = GameManager()
        gm.start_game()
        
        # Manual resolution to compare
        resolution = CardResolver.resolve(A, H, [], [])
        r = gm.play_round(A, H)
        
        assert r.damage_a == resolution.damage_a
        assert r.damage_b == resolution.damage_b
        assert gm.player_a.hp == 3 - resolution.damage_a
        assert gm.player_b.hp == 3 - resolution.damage_b

    def test_resolver_details_in_round_result(self):
        """Resolution details are passed through to RoundResult."""
        gm = GameManager()
        gm.start_game()
        r = gm.play_round(A, D)
        assert len(r.details) > 0
        assert any("Defense" in d or "block" in d.lower() for d in r.details)

    def test_custom_hand_and_resolver_consistency(self):
        """Custom hands don't break the resolver-game_manager contract."""
        gm = GameManager()
        gm.start_game()
        gm.player_a.hand = [A, D, H, C]
        gm.player_b.hand = [A, D, H, C]
        
        # Play each card type and verify HP changes match resolver output
        r1 = gm.play_round(A, D)
        resolution1 = CardResolver.resolve(A, D, [], [])
        assert r1.damage_a == resolution1.damage_a
        assert r1.damage_b == resolution1.damage_b
        
        r2 = gm.play_round(H, C)
        resolution2 = CardResolver.resolve(H, C, [A], [D])
        assert r2.damage_a == resolution2.damage_a
        assert r2.damage_b == resolution2.damage_b


# ---------------------------------------------------------------------------
# Round sequence boundary tests
# ---------------------------------------------------------------------------

class TestRoundSequenceBoundaries:
    """Test boundary conditions across multiple rounds."""

    def test_max_damage_never_exceeds_3(self):
        """In a standard game, max damage per round is 3 (consecutive ATK ×3)."""
        gm = GameManager()
        gm.start_game()
        gm.player_a.history = [A, A]  # Consecutive=2, playing ATK → ×3
        gm.player_b.history = [A, A]  # Consecutive=2, playing ATK → ×3
        r = gm.play_round(A, A)
        assert r.damage_a <= 3
        assert r.damage_b <= 3

    def test_consecutive_streak_persists_across_rounds(self):
        """Consecutive attack count carries over correctly between plays."""
        gm = GameManager()
        gm.start_game()
        # Set HP high enough to survive multiple rounds of increasing damage
        gm.player_a.hp = 10
        gm.player_b.hp = 10
        gm.player_a.hand = [A] * 7
        gm.player_b.hand = [A] * 7

        r1 = gm.play_round(A, A)  # ×1 each
        assert r1.damage_a == 1
        assert r1.damage_b == 1
        r2 = gm.play_round(A, A)  # ×2 each
        assert r2.damage_a == 2
        assert r2.damage_b == 2
        r3 = gm.play_round(A, A)  # ×3 each
        assert r3.damage_a == 3
        assert r3.damage_b == 3
        r4 = gm.play_round(A, A)  # ×4 each
        assert r4.damage_a == 4
        assert r4.damage_b == 4
        # HP after 4 rounds: 10 - (1+2+3+4) = 0
        assert gm.player_a.hp == 0
        assert gm.player_b.hp == 0
        assert r4.is_game_over is True

    def test_defense_streak_resets_on_non_defense(self):
        """Playing a non-Defense card breaks the consecutive-Defense streak."""
        gm = GameManager()
        gm.start_game()
        gm.player_a.hand = [D, D, A, D, H, D, D]
        gm.player_b.hand = [H] * 7  # B always plays HS
        
        gm.play_round(D, H)  # R1: no prev → A takes 1 (HP 2)
        assert gm.player_a.hp == 2
        gm.play_round(D, H)  # R2: consecutive DEF → A takes 0
        assert gm.player_a.hp == 2
        gm.play_round(A, H)  # R3: A plays ATK, breaks streak
        assert gm.player_a.hp == 1  # A takes 1 from HS, B takes 1 from ATK
        gm.play_round(D, H)  # R4: Single DEF, no prev DEF → A takes 1
        assert gm.player_a.hp == 0  # A dies


# ---------------------------------------------------------------------------
# get_state() integration tests
# ---------------------------------------------------------------------------

class TestGetStateIntegration:
    """get_state() returns correct information after complex game flows."""

    def test_get_state_mid_game(self):
        """get_state reflects accurate state mid-game."""
        gm = GameManager()
        gm.start_game()
        gm.play_round(A, D)
        gm.play_round(H, C)
        
        state = gm.get_state()
        assert state["round"] == 3
        assert state["is_over"] is False
        assert state["winner"] is None
        assert state["players"]["A"]["hp"] == 2  # 3-1 from HS vs CTR
        assert state["players"]["B"]["hp"] == 2  # 3-1 from HS vs CTR
        assert len(state["players"]["A"]["hand"]) == 5  # 7-2
        assert state["players"]["A"]["history"] == ["attack", "heavy"]
        assert state["players"]["B"]["history"] == ["defense", "counter"]

    def test_get_state_after_game_over(self):
        """get_state returns correct winner info after game ends."""
        gm = GameManager()
        gm.start_game()
        gm.player_a.hp = 1
        r = gm.play_round(A, A)
        assert r.is_game_over is True
        
        state = gm.get_state()
        assert state["is_over"] is True
        assert state["winner"] == "B"
        assert state["players"]["A"]["hp"] == 0
